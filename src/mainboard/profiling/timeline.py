"""Device timeline occupancy: how much of the wall clock the GPU was actually working.

A kernel ranking answers which kernel costs most. It cannot answer whether the device was
busy at all, and a pipeline that looks kernel-bound is often idle between launches waiting
on the host. Merging the observed kernel and copy intervals gives the busy union, and
everything the span covers but the union does not is the device sitting still.

The gaps are reported with the activity on each side, because an idle window between two
kernels of the same stage means something different from one that straddles a stage
boundary or a host synchronisation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.table import Table

from ..models.base import FrozenModel

if TYPE_CHECKING:
    from collections.abc import Sequence

    from rich.console import RenderableType

    from .trace import KernelTrace, MemcpyTrace

NANOSECONDS_PER_MS = 1_000_000


class DeviceGap(FrozenModel):
    """One window in which no kernel or copy was executing.

    after: name of the activity that ended the busy run before this gap.
    before: name of the activity that starts the next busy run.
    """

    start_ns: int
    end_ns: int
    after: str = ""
    before: str = ""

    @property
    def duration_ms(self) -> float:
        """Return the idle duration in milliseconds."""
        return (self.end_ns - self.start_ns) / NANOSECONDS_PER_MS


class DeviceTimeline(FrozenModel):
    """Busy and idle accounting over the observed device activity.

    span_ns: first activity start to last activity end.
    busy_ns: total time covered by the union of activity intervals, so overlapping
        kernels are counted once rather than summed.
    gaps: idle windows, longest first.
    """

    span_ns: int = 0
    busy_ns: int = 0
    activities: int = 0
    gaps: tuple[DeviceGap, ...] = ()

    @property
    def idle_ns(self) -> int:
        """Return the time inside the span with nothing executing."""
        return max(self.span_ns - self.busy_ns, 0)

    @property
    def occupancy_pct(self) -> float:
        """Return the share of the span during which the device was executing."""
        return 100.0 * self.busy_ns / self.span_ns if self.span_ns else 0.0

    @classmethod
    def from_traces(
        cls,
        kernels: Sequence[KernelTrace],
        memcpys: Sequence[MemcpyTrace] = (),
        *,
        top_gaps: int = 10,
        min_gap_ns: int = 10_000,
    ) -> DeviceTimeline:
        """Build the timeline from observed activity.

        top_gaps: how many idle windows to retain, longest first.
        min_gap_ns: windows shorter than this are launch jitter rather than a stall and
            are counted in the idle total but not listed.
        """
        spans = [(k.start_ns, k.end_ns, k.name) for k in kernels if k.end_ns > k.start_ns]
        spans += [(m.start_ns, m.end_ns, "memcpy") for m in memcpys if m.end_ns > m.start_ns]
        if not spans:
            return cls()
        spans.sort()

        busy = 0
        gaps: list[DeviceGap] = []
        run_start, run_end, run_name = spans[0]
        for start, end, name in spans[1:]:
            if start > run_end:
                busy += run_end - run_start
                if start - run_end >= min_gap_ns:
                    gaps.append(
                        DeviceGap(start_ns=run_end, end_ns=start, after=run_name, before=name)
                    )
                run_start, run_end, run_name = start, end, name
                continue
            if end > run_end:
                run_end, run_name = end, name
        busy += run_end - run_start

        # The span runs from the earliest start to the latest end across every activity, which
        # is not necessarily the last one sorted by start: a long kernel can still be running
        # when a shorter, later-starting one finishes first.
        return cls(
            span_ns=max(end for _, end, _ in spans) - spans[0][0],
            busy_ns=busy,
            activities=len(spans),
            gaps=tuple(sorted(gaps, key=lambda g: g.end_ns - g.start_ns, reverse=True)[:top_gaps]),
        )

    def __rich__(self) -> RenderableType:
        """Render the occupancy summary and the longest idle windows."""
        table = Table(title="device timeline", title_style="bold")
        table.add_column("metric")
        table.add_column("value", justify="right")
        table.add_row("activities", f"{self.activities}")
        table.add_row("span", f"{self.span_ns / 1e6:.2f} ms")
        table.add_row("busy", f"{self.busy_ns / 1e6:.2f} ms")
        table.add_row("idle", f"{self.idle_ns / 1e6:.2f} ms")
        table.add_row("occupancy", f"{self.occupancy_pct:.1f}%")
        for gap in self.gaps:
            table.add_row(f"gap {gap.after} → {gap.before}", f"{gap.duration_ms:.2f} ms")
        return table

    def __str__(self) -> str:
        """Return a one-line summary."""
        return (
            f"span {self.span_ns / 1e6:.1f} ms, busy {self.busy_ns / 1e6:.1f} ms, "
            f"occupancy {self.occupancy_pct:.1f}%, {len(self.gaps)} listed gaps"
        )
