import subprocess
from pathlib import Path

import pytest

from mainboard import cli
from mainboard.profiling.counters import (
    KernelCounterDelta,
    KernelCounters,
    KernelDivergence,
    KernelVerdict,
)
from mainboard.profiling.result import Profile
from mainboard.profiling.target import Target
from mainboard.providers.nvidia.counters import NcuCsvParser, NcuProfiler

_FIXTURES = Path(__file__).parent / "fixtures"


def capture(name: str) -> Profile:
    """Parse one recorded raw Nsight Compute fixture."""
    return NcuCsvParser().parse((_FIXTURES / name).read_text())


def verdict_metrics(
    *,
    ipc: float = 1.0,
    occupancy: float = 50.0,
    stall: float = 1.0,
    sm: float = 20.0,
    l1: float = 20.0,
    l2: float = 20.0,
    dram: float = 20.0,
) -> dict[str, float]:
    """Build complete synthetic evidence for verdict threshold tests."""
    metrics = {
        "gpu__time_duration.sum": 1_000.0,
        "gpc__cycles_elapsed.avg.per_second": 2e9,
        "sm__inst_executed.avg.per_cycle_active": ipc,
        "sm__warps_active.avg.pct_of_peak_sustained_active": occupancy,
        "sm__throughput.avg.pct_of_peak_sustained_elapsed": sm,
        "l1tex__throughput.avg.pct_of_peak_sustained_elapsed": l1,
        "lts__throughput.avg.pct_of_peak_sustained_elapsed": l2,
        "dram__throughput.avg.pct_of_peak_sustained_elapsed": dram,
    }
    metrics.update(
        {
            f"smsp__warp_issue_stalled_{name}_per_warp_active.pct": stall
            for name in (
                "long_scoreboard",
                "wait",
                "branch_resolving",
                "no_instruction",
                "short_scoreboard",
                "not_selected",
                "mio_throttle",
                "lg_throttle",
            )
        }
    )
    return metrics


def test_raw_csv_parses_and_aggregates_derived_counter_evidence() -> None:
    """Real shaped raw rows become one aggregate with persisted derivations."""
    profile = capture("ncu_raw_ada.csv")
    vector = next(row for row in profile.counters if row.demangled_name.startswith("vector_load"))
    assert profile.host == "gold"
    assert profile.device == "NVIDIA GeForce RTX 4090"
    assert vector.calls == 2
    assert vector.duration_ns == 3_000.0
    assert vector.achieved_clock_hz == 2e9
    assert vector.cycles == 6_000.0
    assert vector.issue_utilization_pct == 60.0
    assert vector.sectors_per_request == 4.0
    assert vector.stall_total_pct == 100.0
    assert vector.stall_long_scoreboard_share_pct == 40.0
    assert vector.theoretical_occupancy_pct == 75.0
    assert vector.occupancy_limit_shared_memory == 4.0
    assert vector.verdict is KernelVerdict.ISSUE_BOUND


@pytest.mark.parametrize(
    ("metrics", "expected"),
    (
        (verdict_metrics(dram=80.0), KernelVerdict.BANDWIDTH_BOUND),
        (verdict_metrics(ipc=2.4), KernelVerdict.ISSUE_BOUND),
        (verdict_metrics(stall=4.0), KernelVerdict.LATENCY_BOUND),
        (verdict_metrics(), KernelVerdict.CLOCK_BOUND),
    ),
)
def test_kernel_verdict_thresholds(metrics: dict[str, float], expected: KernelVerdict) -> None:
    """Each documented threshold selects its corresponding verdict."""
    assert KernelCounters.from_metrics("kernel", metrics).verdict is expected


def test_incomplete_counter_evidence_has_unknown_verdict() -> None:
    """A duration alone cannot support a hardware bottleneck verdict."""
    row = KernelCounters.from_metrics("kernel", {"gpu__time_duration.sum": 100.0})
    assert row.verdict is KernelVerdict.UNKNOWN


def test_aggregate_rejects_an_empty_sample_list() -> None:
    """There is no kernel to aggregate without at least one launch."""
    with pytest.raises(ValueError, match="at least one kernel counter sample"):
        KernelCounters.aggregate([])


def test_aggregate_rejects_samples_naming_different_kernels() -> None:
    """Mixing two kernels' launches would silently blend unrelated evidence."""
    foo = KernelCounters.from_metrics("foo_kernel", verdict_metrics(), demangled_name="foo()")
    bar = KernelCounters.from_metrics("bar_kernel", verdict_metrics(), demangled_name="bar()")
    with pytest.raises(ValueError, match="must share one demangled name"):
        KernelCounters.aggregate([foo, bar])


def test_weighted_falls_back_to_call_weighting_without_any_duration() -> None:
    """A sample missing its duration still contributes, weighted by how often it ran."""
    light = KernelCounters.from_metrics("k", {}).model_copy(update={"ipc": 2.0, "calls": 1})
    heavy = KernelCounters.from_metrics("k", {}).model_copy(update={"ipc": 4.0, "calls": 3})
    assert KernelCounters.weighted((light, heavy), lambda sample: sample.ipc) == 3.5


def test_weighted_is_zero_without_duration_or_calls() -> None:
    """Neither weight is available, so the metric reports absent instead of dividing by zero."""
    empty = KernelCounters.from_metrics("k", {}).model_copy(update={"calls": 0})
    assert KernelCounters.weighted((empty,), lambda sample: sample.ipc) == 0.0


def test_relative_change_from_a_zero_baseline() -> None:
    """A zero baseline cannot be divided into, so the sign of `current` alone must decide."""
    assert KernelCounterDelta.relative_change(0.0, 0.0) == 0.0
    assert KernelCounterDelta.relative_change(0.0, 5.0) == 100.0


def test_cross_host_diff_separates_clock_and_architecture_signatures() -> None:
    """Equal cycles isolate clock loss while changed cycles identify different work."""
    baseline = capture("ncu_raw_ada.csv")
    current = capture("ncu_raw_gh200.csv")
    diff = current.diff(baseline)
    vector = next(row for row in diff.kernels if row.name.startswith("vector_load"))
    architecture = next(row for row in diff.kernels if row.name.startswith("architecture_kernel"))
    assert diff.baseline_host == "gold"
    assert diff.current_host == "miyabi"
    assert vector.baseline_cycles == vector.current_cycles == 6_000.0
    assert vector.clock_change_pct == -25.0
    assert vector.divergence is KernelDivergence.CLOCK_BOUND
    assert architecture.cycle_change_pct == 20.0
    assert architecture.divergence is KernelDivergence.ARCHITECTURAL


def test_diff_aligns_distinct_symbols_by_demangled_name() -> None:
    """Raw symbol spelling does not prevent cross-host alignment."""
    metrics = verdict_metrics()
    baseline = KernelCounters.from_metrics("_Z3foov", metrics, demangled_name="foo()")
    current = KernelCounters.from_metrics("foo_kernel", metrics, demangled_name="foo()")
    diff = Profile(counters=(current,)).diff(Profile(counters=(baseline,)))
    assert len(diff.kernels) == 1
    assert diff.kernels[0].name == "foo()"
    assert diff.kernels[0].divergence is KernelDivergence.EQUAL


def test_missing_kernel_is_explicit_in_cross_host_diff() -> None:
    """A kernel present on only one host is not labeled as an architecture change."""
    row = KernelCounters.from_metrics("only_here", verdict_metrics())
    diff = Profile(counters=(row,)).diff(Profile())
    assert diff.kernels[0].divergence is KernelDivergence.MISSING


def test_incomplete_cycle_evidence_is_not_called_architectural() -> None:
    """Missing clock evidence remains incomplete instead of implying changed work."""
    baseline = KernelCounters.from_metrics("kernel", {"gpu__time_duration.sum": 100.0})
    current = KernelCounters.from_metrics("kernel", {"gpu__time_duration.sum": 200.0})
    diff = Profile(counters=(current,)).diff(Profile(counters=(baseline,)))
    assert diff.kernels[0].divergence is KernelDivergence.INCOMPLETE


def test_show_diff_renders_kernel_counters_without_any_region_table(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A diff carrying only counter evidence skips the region table, not the kernel one."""
    baseline = KernelCounters.from_metrics("kernel", verdict_metrics())
    current = KernelCounters.from_metrics("kernel", verdict_metrics(ipc=2.0))
    diff = Profile(counters=(current,)).diff(Profile(counters=(baseline,)))
    assert diff.rows == ()
    diff.show(color=False)
    out = capsys.readouterr().out
    assert "profile diff" not in out
    assert "kernel counter diff" in out
    assert "kernel" in out


def test_counter_profile_round_trips_and_renders(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Counter evidence survives storage and appears in the shared profile view."""
    profile = capture("ncu_raw_ada.csv")
    path = tmp_path / "capture.mbprof"
    profile.save(path)
    loaded = Profile.load(path)
    assert loaded.counters[0].cycles == profile.counters[0].cycles
    loaded.show(color=False)
    assert "kernel counters" in capsys.readouterr().out


def test_ncu_command_contains_every_curated_metric_and_raw_csv_flags() -> None:
    """The provider owns the complete reproducible Nsight Compute command."""
    ncu = Path("/opt/nvidia/ncu")
    executable = Path("/env/python")
    command = NcuProfiler(
        ncu=ncu,
        executable=executable,
        launch_skip=2,
        launch_count=3,
    ).command(Target.resolve("package.train"))
    joined = " ".join(command)
    assert command[:5] == (str(ncu), "--csv", "--page", "raw", "--replay-mode")
    assert "--kernel-name-base demangled" in joined
    assert "gpu__time_duration.sum" in joined
    assert "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct" in joined
    assert "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum" in joined
    assert "dram__throughput.avg.pct_of_peak_sustained_elapsed" in joined
    assert command[-3:] == (str(executable), "-m", "package.train")


def test_ncu_provider_executes_once_and_parses_stdout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The external pass is bounded and feeds its CSV back into `Profile`."""
    csv_text = (_FIXTURES / "ncu_raw_ada.csv").read_text()
    seen: list[tuple[str, ...]] = []

    def fake_run(
        command: tuple[str, ...],
        *,
        check: bool,
        capture_output: bool,
        text: bool,
        timeout: float | None,
    ) -> subprocess.CompletedProcess[str]:
        assert check and capture_output and text and timeout == 4.0
        seen.append(command)
        return subprocess.CompletedProcess(command, 0, stdout=csv_text, stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    profile = NcuProfiler(timeout=4.0).run("work.py")
    assert len(seen) == 1
    assert profile.host == "gold"


def test_raw_csv_without_header_fails_clearly() -> None:
    """Malformed profiler output raises at the parsing boundary."""
    with pytest.raises(ValueError, match="header not found"):
        NcuCsvParser().parse("==ERROR== permission denied")


def test_metric_value_skips_unparsable_numbers() -> None:
    """`n/a` and other non-numeric readings are absent evidence, not a crash."""
    assert NcuCsvParser.metric_value("n/a") is None
    assert NcuCsvParser.metric_value("1,234.5") == 1234.5


def test_parse_skips_a_row_missing_its_metric_name() -> None:
    """A row ncu left blank contributes nothing rather than a bogus counter."""
    header = (
        '"ID","Process ID","Process Name","Host Name","Kernel Name","Context","Stream",'
        '"Block Size","Grid Size","Device","CC","Section Name","Metric Name","Metric Unit",'
        '"Metric Value"\n'
    )
    blank = (
        '"0","1","python","gold","k(float*)","1","1","1,1,1","1,1,1","GPU","8.9","Raw","","",""\n'
    )
    good = (
        '"0","1","python","gold","k(float*)","1","1","1,1,1","1,1,1","GPU","8.9","Raw",'
        '"gpu__time_duration.sum","nsecond","1,000"\n'
    )
    profile = NcuCsvParser().parse(header + blank + good)
    assert profile.counters[0].name == "k(float*)"
    assert profile.counters[0].duration_ns == 1000.0


def test_capture_wraps_a_parse_failure_with_the_raw_ncu_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A header-less run means ncu never produced counters, and why is in its own output."""

    def fake_run(
        command: tuple[str, ...],
        *,
        check: bool,
        capture_output: bool,
        text: bool,
        timeout: float | None,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            command, 0, stdout="", stderr="==ERROR== permission denied"
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(ValueError, match="Nsight Compute said"):
        NcuProfiler().run("work.py")


class FakeNcuProfiler:
    """CLI stand in that records construction without invoking a profiler."""

    configured: tuple[Path, Path, float | None, int, int] | None = None

    def __init__(
        self,
        *,
        ncu: Path,
        executable: Path,
        timeout: float | None,
        launch_skip: int,
        launch_count: int,
    ) -> None:
        type(self).configured = (ncu, executable, timeout, launch_skip, launch_count)

    def run(self, target: str, *, args: tuple[str, ...]) -> Profile:
        return Profile(
            counters=(KernelCounters.from_metrics(target, verdict_metrics()),),
        )


def test_profile_counters_cli_path_uses_ncu_provider(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The counters command is a first class child of `mainboard profile`."""
    monkeypatch.setattr(cli, "NcuProfiler", FakeNcuProfiler)
    with pytest.raises(SystemExit) as exit_signal:
        cli.app(
            ["profile", "counters", "work.py", "--ncu", "/opt/ncu", "--no-color"],
            exit_on_error=False,
        )
    assert exit_signal.value.code == 0
    assert FakeNcuProfiler.configured is not None
    assert FakeNcuProfiler.configured[0] == Path("/opt/ncu")
    assert "kernel counters" in capsys.readouterr().out
