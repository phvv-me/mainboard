from dataclasses import dataclass

import pytest

from mainboard.profile import (
    Activity,
    ActivityRecord,
    BottleneckReport,
    CallbackSession,
    KernelTrace,
    MemcpyTrace,
    Profile,
    RegionWindow,
    TraceCollector,
    busy_ns,
)

from .support import kernel, traced_profile


@dataclass
class FakeKernelActivity:
    """A CUPTI CONCURRENT_KERNEL record as the buffer hands it over."""

    name: str
    start: int
    end: int
    kind: int = 10
    grid_x: int = 8
    grid_y: int = 1
    grid_z: int = 1
    block_x: int = 128
    block_y: int = 2
    block_z: int = 1
    static_shared_memory: int = 512
    dynamic_shared_memory: int = 256
    registers_per_thread: int = 40


@dataclass
class FakeMemcpyActivity:
    """A CUPTI MEMCPY record as the buffer hands it over."""

    copy_kind: int
    start: int
    end: int
    kind: int = 1
    bytes: int = 0


@pytest.mark.parametrize(
    ("kinds", "label"),
    [
        (Activity.KERNEL, "kernel"),
        (Activity.KERNEL | Activity.MEMCPY, "default"),
        (Activity(0), "activity"),
    ],
    ids=["one_flag", "a_named_combination", "no_flag_at_all"],
)
def test_activity_labels_a_named_flag_by_name_and_anything_else_generically(
    kinds: Activity, label: str
) -> None:
    assert kinds.label == label


def test_a_cupti_kernel_record_becomes_a_typed_trace() -> None:
    trace = KernelTrace.from_activity(FakeKernelActivity(name="k", start=0, end=1000))
    assert trace.grid == "8x1x1"
    assert trace.block == "128x2x1"
    assert trace.shared_mem == 768  # 512 static + 256 dynamic
    assert trace.registers == 40
    assert trace.duration_us == 1.0
    assert trace.occupancy_pct == 25.0  # 256 threads over the 1024 hardware max


@pytest.mark.parametrize(
    ("copy_kind", "end", "moved", "label", "bandwidth_gbps"),
    [(1, 1000, 2000, "HtoD", 2.0), (99, 0, 0, "kind_99", 0.0)],
    ids=["a_known_direction", "an_unknown_direction_with_no_measured_time"],
)
def test_a_cupti_memcpy_record_maps_its_direction_and_yields_a_bandwidth(
    copy_kind: int, end: int, moved: int, label: str, bandwidth_gbps: float
) -> None:
    """A direction code outside the table keeps its number rather than reading as unknown."""
    trace = MemcpyTrace.from_activity(
        FakeMemcpyActivity(copy_kind=copy_kind, start=0, end=end, bytes=moved)
    )
    assert trace.kind == label
    assert trace.bandwidth_gbps == bandwidth_gbps
    assert trace.duration_ns == end


@pytest.mark.parametrize(
    ("block", "threads"),
    [("", 1), ("16xNx2", 32), ("256x1x1", 256)],
    ids=["no_shape_at_all", "a_dimension_that_is_not_a_number", "the_cupti_spelling"],
)
def test_threads_per_block_degrades_to_the_dimensions_it_can_parse(
    block: str, threads: int
) -> None:
    assert KernelTrace(block=block).threads_per_block == threads
    assert ActivityRecord(start_ns=10, end_ns=60).duration_ns == 50


def test_the_deep_report_splits_compute_from_copy_and_ranks_the_hot_spots() -> None:
    report = traced_profile().trace_report()
    assert isinstance(report, BottleneckReport)
    assert report.compute_pct > report.memcpy_pct
    assert report.compute_pct + report.memcpy_pct == pytest.approx(100.0)
    assert report.hot_kernels[0].name == "gemm"
    assert report.hot_regions[0].name in {"encode", "decode"}
    assert sum(region.kernel_count for region in report.hot_regions) == 3

    empty = Profile().trace_report()
    assert empty.total_kernel_ns == 0
    assert empty.hot_kernels == ()


@pytest.mark.parametrize(
    ("spans", "busy"),
    [
        ((), 0),
        (((0, 100),), 100),
        (((0, 100), (100, 250)), 250),
        (((0, 100), (40, 60)), 100),
        (((0, 100), (60, 140)), 140),
        (((60, 140), (0, 100)), 140),
        (((0, 100), (200, 260)), 160),
        (((10, 10), (0, 50)), 50),
    ],
    ids=[
        "nothing",
        "one_span",
        "abutting",
        "nested",
        "overlapping",
        "unsorted",
        "disjoint",
        "empty_span_dropped",
    ],
)
def test_device_busy_time_is_a_union_and_never_a_sum(
    spans: tuple[tuple[int, int], ...], busy: int
) -> None:
    """Summed durations measure work, not time, so a share against wall divides the union."""
    assert busy_ns(spans) == busy


def test_the_summed_work_time_can_exceed_the_clock_and_the_report_carries_both() -> None:
    """`recovery_cost`'s 2026-08-29 finding: a summed device time read 1.9-2.1x the CUDA-event
    ground truth at ten launches. Overlap makes the sum larger at any count."""
    profile = Profile(
        kernels=(kernel("a", 100, start_ns=0), kernel("b", 100, start_ns=30)),
        memcpys=(MemcpyTrace(start_ns=80, end_ns=130, bytes_moved=1024),),
    )
    report = profile.trace_report()
    assert report.total_kernel_ns + report.total_memcpy_ns == 250
    assert report.device_busy_ns == 130
    assert report.device_busy_ns < report.total_kernel_ns + report.total_memcpy_ns


@pytest.mark.parametrize(
    ("windows", "attributed"),
    [
        ((RegionWindow(name="r", start_ns=0, end_ns=10, wall_ns=10),), "(outside regions)"),
        (
            (
                RegionWindow(name="inner", start_ns=100, end_ns=300, wall_ns=200),
                RegionWindow(name="outer", start_ns=0, end_ns=1000, wall_ns=1000),
            ),
            "inner",
        ),
    ],
    ids=["a_kernel_in_no_window", "nested_windows"],
)
def test_a_kernel_is_attributed_to_the_narrowest_window_that_contains_it(
    windows: tuple[RegionWindow, ...], attributed: str
) -> None:
    """A kernel in no window is labeled rather than dropped, so unattributed time stays visible."""
    profile = Profile(windows=windows, kernels=(kernel("k", 50, start_ns=150),))
    assert profile.trace_report().hot_regions[0].name == attributed


def test_the_base_collector_and_callback_session_are_safe_noops() -> None:
    """The base refuses only a synchronized window, which would read as a region that ran no
    GPU work."""
    with TraceCollector() as collector:
        collector.flush()
        collector.reset()
    with pytest.raises(RuntimeError, match="unavailable on this backend"):
        collector.checkpoint(Activity.KERNEL)
    with pytest.raises(RuntimeError, match="no synchronized CUDA device"):
        _ = collector.device_index
    assert collector.kernels() == []
    assert collector.memcpys() == []
    assert collector.activities() == []
    assert collector.dropped() == 0
    with CallbackSession() as session:
        assert session.counts() == {}
