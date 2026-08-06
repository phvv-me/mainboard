import csv
import subprocess
import sys
from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path

from ...models.base import Field, FrozenModel, FrozenSequence
from ...profiling.counters import KernelCounters
from ...profiling.result import Profile
from ...profiling.target import Target

_METRICS = (
    "gpu__time_duration.sum",
    "gpc__cycles_elapsed.avg.per_second",
    "sm__inst_executed.avg.per_cycle_active",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "launch__occupancy_per_block_size",
    "launch__occupancy_limit_blocks",
    "launch__occupancy_limit_registers",
    "launch__occupancy_limit_shared_mem",
    "launch__occupancy_limit_warps",
    "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
    "smsp__warp_issue_stalled_wait_per_warp_active.pct",
    "smsp__warp_issue_stalled_branch_resolving_per_warp_active.pct",
    "smsp__warp_issue_stalled_no_instruction_per_warp_active.pct",
    "smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct",
    "smsp__warp_issue_stalled_not_selected_per_warp_active.pct",
    "smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct",
    "smsp__warp_issue_stalled_lg_throttle_per_warp_active.pct",
    "l1tex__t_sector_hit_rate.pct",
    "lts__t_sector_hit_rate.pct",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
    "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "l1tex__throughput.avg.pct_of_peak_sustained_elapsed",
    "lts__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
)
_DURATION = "gpu__time_duration.sum"
_CLOCK = "gpc__cycles_elapsed.avg.per_second"
_DURATION_FACTORS = {
    "ns": 1.0,
    "nsecond": 1.0,
    "nseconds": 1.0,
    "nanosecond": 1.0,
    "nanoseconds": 1.0,
    "us": 1_000.0,
    "usecond": 1_000.0,
    "useconds": 1_000.0,
    "microsecond": 1_000.0,
    "microseconds": 1_000.0,
    "ms": 1_000_000.0,
    "msecond": 1_000_000.0,
    "mseconds": 1_000_000.0,
    "millisecond": 1_000_000.0,
    "milliseconds": 1_000_000.0,
    "s": 1_000_000_000.0,
    "second": 1_000_000_000.0,
    "seconds": 1_000_000_000.0,
}
_CLOCK_FACTORS = {
    "cycle/ns": 1e9,
    "cycle/nsecond": 1e9,
    "cycle/nanosecond": 1e9,
    "cycle/us": 1e6,
    "cycle/usecond": 1e6,
    "cycle/microsecond": 1e6,
    "cycle/ms": 1e3,
    "cycle/msecond": 1e3,
    "cycle/millisecond": 1e3,
    "cycle/s": 1.0,
    "cycle/second": 1.0,
    "hz": 1.0,
    "khz": 1e3,
    "mhz": 1e6,
    "ghz": 1e9,
}


@dataclass(slots=True)
class LaunchMetrics:
    """Mutable parse buffer for one kernel launch in the raw CSV."""

    name: str
    host: str
    device: str
    values: dict[str, float] = field(default_factory=dict)


class NcuCsvParser:
    """Parse Nsight Compute raw CSV into the shared immutable `Profile`."""

    def parse(self, text: str) -> Profile:
        """Parse one `ncu --csv --page raw` stream."""
        launches: dict[tuple[str, str, str, str, str], LaunchMetrics] = {}
        for row in self.rows(text):
            metric = row.get("Metric Name", "").strip()
            value = self.metric_value(row.get("Metric Value", ""))
            if not metric or value is None:
                continue
            name = row.get("Kernel Name", "").strip()
            key = (
                row.get("Host Name", "").strip(),
                row.get("Process ID", "").strip(),
                row.get("ID", "").strip(),
                name,
                row.get("Device", "").strip(),
            )
            launch = launches.setdefault(
                key,
                LaunchMetrics(name=name, host=key[0], device=key[4]),
            )
            launch.values[metric] = self.normalize(metric, row.get("Metric Unit", ""), value)
        samples = tuple(
            KernelCounters.from_metrics(launch.name, launch.values)
            for launch in launches.values()
            if launch.name
        )
        first = next(iter(launches.values()), None)
        return Profile(
            host=first.host if first is not None else "",
            device=first.device if first is not None else "",
            counters=KernelCounters.aggregate_by_name(samples),
        )

    @staticmethod
    def rows(text: str) -> tuple[dict[str, str], ...]:
        """Return data rows after the real Nsight Compute header."""
        header: tuple[str, ...] | None = None
        parsed: list[dict[str, str]] = []
        for fields in csv.reader(StringIO(text)):
            cleaned = tuple(field.strip().lstrip("\ufeff") for field in fields)
            if header is None:
                if "Metric Name" in cleaned and "Metric Value" in cleaned:
                    header = cleaned
                continue
            if "Metric Name" in cleaned and "Metric Value" in cleaned:
                header = cleaned
                continue
            if len(cleaned) != len(header):
                continue
            parsed.append(dict(zip(header, cleaned, strict=True)))
        if header is None:
            raise ValueError("Nsight Compute CSV header not found")
        return tuple(parsed)

    @staticmethod
    def metric_value(raw: str) -> float | None:
        """Parse a locale independent CSV number and skip unavailable values."""
        try:
            return float(raw.strip().replace(",", ""))
        except ValueError:
            return None

    @staticmethod
    def normalize(metric: str, unit: str, value: float) -> float:
        """Normalize duration to nanoseconds and achieved clock to hertz."""
        normalized_unit = unit.strip().casefold().replace(" ", "")
        if metric == _DURATION:
            return value * _DURATION_FACTORS.get(normalized_unit, 1.0)
        if metric == _CLOCK:
            return value * _CLOCK_FACTORS.get(normalized_unit, 1.0)
        return value


class NcuProfiler(FrozenModel):
    """Typed command interface for the replay based NVIDIA counter pass."""

    ncu: Path = Path("ncu")
    executable: Path = Path(sys.executable)
    timeout: float = Field(default=600.0, gt=0)
    launch_skip: int = Field(default=0, ge=0)
    launch_count: int = Field(default=0, ge=0)
    additional_metrics: FrozenSequence[str] = ()

    def run(
        self,
        target: str,
        *,
        module: bool | None = None,
        args: tuple[str, ...] = (),
    ) -> Profile:
        """Launch one target under Nsight Compute and return its `Profile`."""
        return self.capture(Target.resolve(target, module=module, args=args))

    def capture(self, target: Target) -> Profile:
        """Execute one bounded counter pass for a resolved target."""
        completed = subprocess.run(
            self.command(target),
            check=True,
            capture_output=True,
            text=True,
            timeout=self.timeout,
        )
        text = "\n".join((completed.stdout, completed.stderr))
        try:
            return NcuCsvParser().parse(text)
        except ValueError as error:
            # The header being absent means ncu never produced counters, and the reason is in
            # its own output, so a parse error that hides that output costs a debugging session.
            tail = "\n".join(line for line in text.splitlines() if line.strip())[-2000:]
            raise ValueError(f"{error}; Nsight Compute said:\n{tail}") from error

    def command(self, target: Target) -> tuple[str, ...]:
        """Build the deterministic raw CSV command for one target."""
        metrics = tuple(dict.fromkeys((*_METRICS, *self.additional_metrics)))
        command = [
            str(self.ncu),
            "--csv",
            "--page",
            "raw",
            "--replay-mode",
            "kernel",
            "--kernel-name-base",
            "demangled",
            "--target-processes",
            "all",
            "--metrics",
            ",".join(metrics),
        ]
        if self.launch_skip:
            command.extend(("--launch-skip", str(self.launch_skip)))
        if self.launch_count:
            command.extend(("--launch-count", str(self.launch_count)))
        command.extend(
            (
                str(self.executable),
                *(("-m",) if target.module else ()),
                target.name,
                *target.args,
            )
        )
        return tuple(command)
