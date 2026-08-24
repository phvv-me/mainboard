from collections.abc import Sequence

from hypothesis import example, given, settings
from hypothesis import strategies as st

from mainboard.profile import DeviceTimeline, KernelTrace, MemcpyTrace

from .support import render

_WINDOWS = st.lists(
    st.tuples(st.integers(min_value=0, max_value=5_000), st.integers(min_value=0, max_value=900)),
    max_size=6,
)


def _timeline(windows: Sequence[tuple[int, int]], *, min_gap_ns: int = 100) -> DeviceTimeline:
    """Build a timeline from (start, duration) pairs, half of them read as copies."""
    kernels = [
        KernelTrace(name=f"k{i}", start_ns=start, end_ns=start + duration)
        for i, (start, duration) in enumerate(windows)
        if i % 2 == 0
    ]
    memcpys = [
        MemcpyTrace(start_ns=start, end_ns=start + duration)
        for i, (start, duration) in enumerate(windows)
        if i % 2
    ]
    return DeviceTimeline.from_traces(kernels, memcpys, min_gap_ns=min_gap_ns, top_gaps=3)


# The pinned examples below carry the branches, so the random budget only needs to add breadth
# and is trimmed from the shared default.
@settings(max_examples=15)
@given(windows=_WINDOWS)
@example(windows=[])  # nothing observed at all
@example(windows=[(100, 0)])  # a zero-duration activity is not an activity
@example(windows=[(0, 1000), (100, 100)])  # a shorter later kernel nested in a longer one
@example(windows=[(0, 1000), (500, 1000)])  # an overlapping copy extends the busy run
@example(windows=[(0, 100), (105, 95)])  # a sub-threshold gap counts as idle but is not listed
@example(windows=[(0, 100), (200, 100)])  # a gap at the threshold is listed
def test_the_timeline_partitions_its_span_into_busy_and_idle(
    windows: list[tuple[int, int]],
) -> None:
    """Busy and idle always add back up to the span, and no gap ever ends before it starts.

    Overlapping activity is counted once rather than summed, so busy can never exceed the
    span and occupancy stays a percentage. Gaps come back longest first, bounded by
    `top_gaps`, and every listed one is at least `min_gap_ns` wide.
    """
    timeline = _timeline(windows)
    observed = [pair for pair in windows if pair[1] > 0]

    assert timeline.activities == len(observed)
    assert timeline.busy_ns + timeline.idle_ns == timeline.span_ns
    assert 0 <= timeline.busy_ns <= timeline.span_ns
    assert 0.0 <= timeline.occupancy_pct <= 100.0
    if observed:
        assert timeline.span_ns == max(start + dur for start, dur in observed) - min(
            start for start, _ in observed
        )
    else:
        assert timeline == DeviceTimeline()

    durations = [gap.end_ns - gap.start_ns for gap in timeline.gaps]
    assert len(durations) <= 3
    assert durations == sorted(durations, reverse=True)
    assert all(duration >= 100 for duration in durations)
    assert all(
        gap.duration_ms == duration / 1e6
        for gap, duration in zip(timeline.gaps, durations, strict=True)
    )


def test_gaps_are_listed_longest_first_with_the_activities_around_them() -> None:
    """A listed gap names what ended the busy run before it and what starts the next one."""
    windows = [(0, 100), (1_100, 100), (1_300, 100), (3_400, 100)]
    timeline = DeviceTimeline.from_traces(
        [
            KernelTrace(name=name, start_ns=start, end_ns=start + dur)
            for name, (start, dur) in zip("abcd", windows, strict=True)
        ],
        min_gap_ns=50,
        top_gaps=2,
    )
    assert [gap.end_ns - gap.start_ns for gap in timeline.gaps] == [2000, 1000]
    assert [(gap.after, gap.before) for gap in timeline.gaps] == [("c", "d"), ("a", "b")]


def test_the_timeline_renders_its_metrics_and_its_gaps() -> None:
    """The table lists occupancy and each idle window, and `str` is the one-line summary."""
    timeline = _timeline([(0, 100), (200, 100)], min_gap_ns=50)
    text = render(timeline)
    assert "device timeline" in text
    assert "occupancy" in text
    assert "gap k0 → memcpy" in text
    assert "occupancy" in str(timeline)
    assert "busy" in str(timeline)
