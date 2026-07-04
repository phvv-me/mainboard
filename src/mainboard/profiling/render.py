"""Rich rendering of profiles and diffs — the human-facing view of the results.

Kept separate from :mod:`.result` so the result value stays pure data; ``Profile.show``
and ``ProfileDiff.show`` delegate here. :func:`region_text` is the no-rich fallback.
"""

from typing import TYPE_CHECKING

from rich import box
from rich.console import Console, Group, RenderableType
from rich.table import Table

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .models import RegionStat, SpanStat
    from .result import Profile, ProfileDiff
    from .trace import BottleneckReport

_MB = 1024**2


def region_text(stats: Sequence[RegionStat]) -> str:
    """Plain-text per-name table (the fallback when rich output is unwanted)."""
    if not stats:
        return "No regions recorded."
    rows = [f"{'region':<30}{'calls':>6}{'total ms':>10}{'avg ms':>9}{'peak MB':>10}"]
    rows += [
        f"{s.name:<30}{s.calls:>6d}{s.total_ms:>10.2f}{s.avg_ms:>9.2f}"
        f"{s.peak_memory_bytes / _MB:>10.1f}"
        for s in stats
    ]
    return "\n".join(rows)


def _region_table(stats: Sequence[RegionStat]) -> Table:
    gpu = any(s.max_util_pct or s.max_power_w for s in stats)
    table = Table(title="regions (by total time)", box=box.ROUNDED, header_style="bold")
    table.add_column("region")
    for name in ("calls", "total ms", "avg ms", "peak MB"):
        table.add_column(name, justify="right")
    if gpu:
        table.add_column("util%", justify="right")
        table.add_column("W", justify="right")
    for s in stats:
        cells = [
            s.name,
            str(s.calls),
            f"{s.total_ms:.2f}",
            f"{s.avg_ms:.2f}",
            f"{s.peak_memory_bytes / _MB:.1f}",
        ]
        if gpu:
            cells += [f"{s.max_util_pct:.0f}", f"{s.max_power_w:.0f}"]
        table.add_row(*cells)
    return table


def _kernel_table(report: BottleneckReport) -> Table:
    title = f"hot kernels  (compute {report.compute_pct:.0f}% / copy {report.memcpy_pct:.0f}%)"
    table = Table(title=title, box=box.ROUNDED, header_style="bold")
    table.add_column("kernel")
    for name in ("calls", "total ms", "share%"):
        table.add_column(name, justify="right")
    for hot in report.hot_kernels:
        table.add_row(
            hot.name[:48], str(hot.calls), f"{hot.total_ns / 1e6:.2f}", f"{hot.share_pct:.1f}"
        )
    return table


def profile_renderable(profile: Profile) -> RenderableType:
    """A rich renderable of a profile: the region table + hot kernels when traced.

    Backs both ``Profile.__rich__`` (so ``print(profile)`` and Jupyter just work) and
    :func:`show_profile`.
    """
    stats = profile.stats()
    if not stats:
        return "No regions recorded."
    if profile.kernels:
        return Group(_region_table(stats), _kernel_table(profile.trace_report()))
    return _region_table(stats)


def show_profile(profile: Profile, *, color: bool = True) -> None:
    """Print the region stats, plus the hot-kernel report when the run was traced."""
    Console(no_color=not color).print(profile_renderable(profile))


def show_diff(diff: ProfileDiff, *, color: bool = True) -> None:
    """Print per-region deltas; green where faster, red where slower."""
    console = Console(no_color=not color)
    table = Table(title="profile diff (baseline → current)", box=box.ROUNDED, header_style="bold")
    table.add_column("region")
    for name in ("baseline ms", "current ms", "Δ ms", "speedup"):
        table.add_column(name, justify="right")
    for row in diff.rows:
        faster = row.speedup > 1.0
        style = "green" if faster else ("red" if row.delta_ms > 0 else None)
        speed = f"{row.speedup:.2f}x" if row.speedup else "—"
        table.add_row(
            row.name,
            f"{row.baseline_ms:.2f}",
            f"{row.current_ms:.2f}",
            f"{row.delta_ms:+.2f}",
            speed,
            style=style,
        )
    console.print(table)


def span_text(stats: Sequence[SpanStat]) -> str:
    """Plain-text per-path table (the fallback when rich output is unwanted)."""
    if not stats:
        return "No spans recorded."
    header = (
        f"{'path':<32}{'calls':>6}{'total ms':>10}{'mean ms':>9}{'p50 ms':>9}"
        f"{'p95 ms':>9}{'max ms':>9}"
    )
    rows = [header]
    rows += [
        f"{s.path:<32}{s.count:>6d}{s.total_ms:>10.2f}{s.mean_ms:>9.2f}"
        f"{s.p50_ms:>9.2f}{s.p95_ms:>9.2f}{s.max_ms:>9.2f}"
        for s in stats
    ]
    return "\n".join(rows)


def _span_table(stats: Sequence[SpanStat]) -> Table:
    table = Table(title="spans (by total time)", box=box.ROUNDED, header_style="bold")
    table.add_column("path")
    for name in ("calls", "total ms", "mean ms", "p50 ms", "p95 ms", "max ms"):
        table.add_column(name, justify="right")
    for s in stats:
        table.add_row(
            s.path,
            str(s.count),
            f"{s.total_ms:.2f}",
            f"{s.mean_ms:.2f}",
            f"{s.p50_ms:.2f}",
            f"{s.p95_ms:.2f}",
            f"{s.max_ms:.2f}",
        )
    return table


def show_spans(stats: Sequence[SpanStat], *, color: bool = True) -> None:
    """Print the per-path span stats as a rich table."""
    if not stats:
        print("No spans recorded.")
        return
    Console(no_color=not color).print(_span_table(stats))
