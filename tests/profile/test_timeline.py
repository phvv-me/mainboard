"""Device timeline occupancy: the busy union, idle gaps, and the rendered summary."""

from rich.console import Console

from mainboard.profile import DeviceTimeline, KernelTrace, MemcpyTrace


def _kernel(name: str, start_ns: int, *, end_ns: int) -> KernelTrace:
    return KernelTrace(name=name, start_ns=start_ns, end_ns=end_ns)


def render(timeline: DeviceTimeline) -> str:
    """Render a timeline to plain text for content assertions."""
    console = Console(no_color=True, width=120, record=True)
    console.print(timeline)
    return console.export_text()


def test_no_activity_yields_an_empty_timeline() -> None:
    timeline = DeviceTimeline.from_traces([])
    assert timeline == DeviceTimeline()
    assert timeline.idle_ns == 0
    assert timeline.occupancy_pct == 0.0


def test_zero_duration_activity_is_discarded() -> None:
    kernels = [_kernel("noop", 100, end_ns=100)]
    timeline = DeviceTimeline.from_traces(kernels)
    assert timeline == DeviceTimeline()


def test_one_kernel_is_fully_busy_with_no_gaps() -> None:
    timeline = DeviceTimeline.from_traces([_kernel("k", 0, end_ns=1000)])
    assert timeline.span_ns == 1000
    assert timeline.busy_ns == 1000
    assert timeline.idle_ns == 0
    assert timeline.occupancy_pct == 100.0
    assert timeline.gaps == ()


def test_a_long_running_kernel_still_bounds_the_span_after_a_shorter_one_finishes() -> None:
    """The span must reach the latest end even when it is not the last one sorted by start."""
    kernels = [_kernel("long", 0, end_ns=1000), _kernel("nested", 100, end_ns=200)]
    timeline = DeviceTimeline.from_traces(kernels)
    assert timeline.span_ns == 1000
    assert timeline.busy_ns == 1000
    assert timeline.idle_ns == 0


def test_an_overlapping_memcpy_extends_the_current_run_without_a_gap() -> None:
    kernels = [_kernel("k", 0, end_ns=1000)]
    memcpys = [MemcpyTrace(start_ns=500, end_ns=1500)]
    timeline = DeviceTimeline.from_traces(kernels, memcpys)
    assert timeline.span_ns == 1500
    assert timeline.busy_ns == 1500
    assert timeline.activities == 2
    assert timeline.gaps == ()


def test_a_gap_shorter_than_the_threshold_is_counted_idle_but_not_listed() -> None:
    kernels = [_kernel("a", 0, end_ns=100), _kernel("b", 105, end_ns=200)]
    timeline = DeviceTimeline.from_traces(kernels, min_gap_ns=10)
    assert timeline.span_ns == 200
    assert timeline.busy_ns == 195
    assert timeline.idle_ns == 5
    assert timeline.gaps == ()


def test_a_gap_at_or_above_the_threshold_is_listed_with_its_neighbours() -> None:
    kernels = [_kernel("a", 0, end_ns=100), _kernel("b", 200, end_ns=300)]
    timeline = DeviceTimeline.from_traces(kernels, min_gap_ns=100)
    assert timeline.busy_ns == 200
    assert timeline.idle_ns == 100
    assert len(timeline.gaps) == 1
    gap = timeline.gaps[0]
    assert (gap.after, gap.before) == ("a", "b")
    assert gap.duration_ms == 100 / 1_000_000


def test_gaps_are_reported_longest_first_and_bounded_by_top_gaps() -> None:
    kernels = [
        _kernel("a", 0, end_ns=100),
        _kernel("b", 1_100, end_ns=1_200),  # 1000 ns gap after a
        _kernel("c", 1_300, end_ns=1_400),  # 100 ns gap after b
        _kernel("d", 3_400, end_ns=3_500),  # 2000 ns gap after c
    ]
    timeline = DeviceTimeline.from_traces(kernels, min_gap_ns=50, top_gaps=2)
    assert [round(g.duration_ms * 1_000_000) for g in timeline.gaps] == [2000, 1000]


def test_rich_render_lists_metrics_and_gaps() -> None:
    kernels = [_kernel("a", 0, end_ns=100), _kernel("b", 200, end_ns=300)]
    timeline = DeviceTimeline.from_traces(kernels, min_gap_ns=50)
    text = render(timeline)
    assert "device timeline" in text
    assert "occupancy" in text
    assert "gap a → b" in text


def test_str_gives_a_one_line_summary() -> None:
    timeline = DeviceTimeline.from_traces(
        [_kernel("a", 0, end_ns=100), _kernel("b", 200, end_ns=300)]
    )
    summary = str(timeline)
    assert "occupancy" in summary
    assert "busy" in summary
