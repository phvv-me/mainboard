"""Per-kernel launch efficiency: waves, tail quantisation and achieved bandwidth.

Occupancy and kernel time answer different questions from "was the grid the right shape".
A grid that is not a whole multiple of the machine's block slots finishes its last wave with
most of the device idle, and that tail is invisible in both a duration ranking and a busy
or idle timeline, because the device reports itself busy while a handful of blocks drain.

Waves are the metric that exposes it. A kernel launched with 1.1 waves spends nearly half
its time at a tenth of the machine, and the fix is a grid shape rather than a faster inner
loop.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from rich.table import Table

from ..models.base import FrozenModel

if TYPE_CHECKING:
    from collections.abc import Sequence

    from rich.console import JustifyMethod, RenderableType

    from .trace import KernelTrace

NS_PER_MS = 1_000_000


def grid_blocks(shape: str) -> int:
    """Return the total extent encoded in a CUPTI launch shape.

    CUPTI writes shapes as ``110x1x1``. Parenthesised and comma-separated spellings are
    accepted too so a backend that formats differently does not silently report zero.
    """
    cleaned = shape.replace("(", " ").replace(")", " ").replace(",", " ").replace("x", " ")
    digits = [int(part) for part in cleaned.split() if part.isdigit()]
    total = 1
    for value in digits:
        total *= value
    return total if digits else 0


def readable(name: str) -> str:
    """Return a kernel name with the Itanium mangling and JIT suffix trimmed off.

    numba appends a long content hash to every symbol, so the tail of a mangled name is
    noise and the informative part is the namespace and function near the front.
    """
    if not name.startswith("_ZN"):
        return name
    body = name[3:]
    parts: list[str] = []
    index = 0
    while index < len(body) and body[index].isdigit():
        width = 0
        while index < len(body) and body[index].isdigit():
            width = width * 10 + int(body[index])
            index += 1
        parts.append(body[index : index + width])
        index += width
    return ".".join(parts[:3]) if parts else name


class KernelEfficiency(FrozenModel):
    """Launch shape and efficiency for one kernel name, aggregated over its calls.

    waves: grid blocks divided by the device's block slots. A whole number means every
        block slot finishes together; a fractional one leaves a partly idle tail wave.
    tail_waste_pct: share of the kernel's time spent in that partial final wave, assuming
        every block costs the same.
    """

    name: str
    calls: int
    total_ms: float
    blocks: int
    block_threads: int
    registers: int
    waves: float
    tail_waste_pct: float

    @classmethod
    def aggregate(
        cls, kernels: Sequence[KernelTrace], sm_count: int, blocks_per_sm: int = 1
    ) -> list[KernelEfficiency]:
        """Rank kernels by total device time, annotating each with its launch shape.

        blocks_per_sm: resident blocks each multiprocessor can hold. Left at one this
            reports the coarsest wave count, which is the one that matters for a grid far
            smaller than the machine.
        """
        slots = max(sm_count * blocks_per_sm, 1)
        groups: defaultdict[str, list[KernelTrace]] = defaultdict(list)
        for kernel in kernels:
            if kernel.end_ns > kernel.start_ns:
                groups[kernel.name].append(kernel)

        rows: list[KernelEfficiency] = []
        for name, traces in groups.items():
            blocks = max(grid_blocks(t.grid) for t in traces)
            waves = blocks / slots
            whole = int(waves)
            partial = waves - whole
            tail = 0.0 if partial == 0 else (1 - partial) / (whole + 1) * 100
            threads = grid_blocks(traces[0].block)
            rows.append(
                cls(
                    name=name,
                    calls=len(traces),
                    total_ms=sum(t.end_ns - t.start_ns for t in traces) / NS_PER_MS,
                    blocks=blocks,
                    block_threads=threads,
                    registers=traces[0].registers,
                    waves=waves,
                    tail_waste_pct=tail,
                )
            )
        return sorted(rows, key=lambda r: r.total_ms, reverse=True)


class EfficiencyReport(FrozenModel):
    """Launch efficiency across every kernel observed, with achieved bandwidth.

    bytes_moved: application-supplied payload size, so achieved bandwidth is reported
        against what the work actually required rather than a guess from the trace.
    """

    rows: tuple[KernelEfficiency, ...] = ()
    sm_count: int = 0
    peak_bandwidth_gbs: float = 0.0
    bytes_moved: int = 0

    @property
    def total_ms(self) -> float:
        """Return total device time across every kernel."""
        return sum(row.total_ms for row in self.rows)

    @property
    def achieved_gbs(self) -> float:
        """Return payload bytes divided by total device time."""
        if not self.total_ms or not self.bytes_moved:
            return 0.0
        return self.bytes_moved / (self.total_ms / 1000) / 1e9

    @property
    def bandwidth_utilisation_pct(self) -> float:
        """Return achieved bandwidth as a share of the device peak."""
        if not self.peak_bandwidth_gbs:
            return 0.0
        return 100.0 * self.achieved_gbs / self.peak_bandwidth_gbs

    @classmethod
    def build(
        cls,
        kernels: Sequence[KernelTrace],
        *,
        sm_count: int,
        peak_bandwidth_gbs: float = 0.0,
        bytes_moved: int = 0,
        blocks_per_sm: int = 1,
        top: int = 12,
    ) -> EfficiencyReport:
        """Build the report from observed kernels and the device's own limits."""
        rows = KernelEfficiency.aggregate(kernels, sm_count, blocks_per_sm)
        return cls(
            rows=tuple(rows[:top]),
            sm_count=sm_count,
            peak_bandwidth_gbs=peak_bandwidth_gbs,
            bytes_moved=bytes_moved,
        )

    def __rich__(self) -> RenderableType:
        """Render the per-kernel launch shapes and the bandwidth summary."""
        title = f"launch efficiency ({self.sm_count} SMs"
        if self.bytes_moved:
            title += (
                f", {self.achieved_gbs:.0f} GB/s achieved"
                f" = {self.bandwidth_utilisation_pct:.1f}% of peak"
            )
        table = Table(title=title + ")", title_style="bold")
        columns: tuple[tuple[str, JustifyMethod], ...] = (
            ("kernel", "left"),
            ("calls", "right"),
            ("total ms", "right"),
            ("blocks", "right"),
            ("thr/blk", "right"),
            ("waves", "right"),
            ("tail %", "right"),
        )
        for column, justify in columns:
            table.add_column(column, justify=justify)
        for row in self.rows:
            table.add_row(
                readable(row.name)[:40],
                f"{row.calls}",
                f"{row.total_ms:.2f}",
                f"{row.blocks}",
                f"{row.block_threads}",
                f"{row.waves:.2f}",
                f"{row.tail_waste_pct:.1f}",
            )
        return table
