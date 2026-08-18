from dataclasses import dataclass

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
)


@dataclass
class _FakeKernelActivity:
    name: str
    start: int
    end: int
    kind: int = 10
    grid_x: int = 1
    grid_y: int = 1
    grid_z: int = 1
    block_x: int = 128
    block_y: int = 1
    block_z: int = 1
    static_shared_memory: int = 0
    dynamic_shared_memory: int = 0
    registers_per_thread: int = 0


@dataclass
class _FakeMemcpyActivity:
    copy_kind: int
    start: int
    end: int
    kind: int = 1
    bytes: int = 0


def _traced_profile() -> Profile:
    """A profile with two regions and kernels/memcpys binned across their windows."""
    return Profile(
        device="dev",
        windows=(
            RegionWindow(name="encode", start_ns=0, end_ns=1000, wall_ns=2_000_000),
            RegionWindow(name="decode", start_ns=1000, end_ns=2000, wall_ns=1_000_000),
        ),
        kernels=(
            KernelTrace(name="gemm", start_ns=0, end_ns=600, grid="8x1x1", block="256x1x1"),
            KernelTrace(name="gemm", start_ns=1000, end_ns=1400),
            KernelTrace(name="relu", start_ns=600, end_ns=700),
        ),
        memcpys=(MemcpyTrace(kind="HtoD", start_ns=0, end_ns=100, bytes_moved=4096),),
        activities=(
            ActivityRecord(kind="runtime", name="cudaLaunchKernel", start_ns=0, end_ns=5),
        ),
    )


def test_activity_label_falls_back_for_compound_flags() -> None:
    """A single flag labels by name; a compound flag has no name, so labels `activity`."""
    assert Activity.KERNEL.label == "kernel"
    assert (Activity.KERNEL | Activity.MEMCPY).label == "default"
    assert Activity(0).label == "activity"


def test_memcpy_trace_from_activity_maps_kind_and_bandwidth() -> None:
    """A CUPTI MEMCPY record maps copy_kind to a label and yields a bandwidth."""
    act = _FakeMemcpyActivity(copy_kind=1, start=0, end=1000, bytes=2000)
    trace = MemcpyTrace.from_activity(act)
    assert trace.kind == "HtoD"
    assert trace.bandwidth_gbps == 2.0
    blank = _FakeMemcpyActivity(copy_kind=99, start=0, end=0)
    unknown = MemcpyTrace.from_activity(blank)
    assert unknown.kind == "kind_99"
    assert unknown.bandwidth_gbps == 0.0  # zero duration


def test_kernel_trace_from_activity_reads_launch_shape() -> None:
    """A CUPTI CONCURRENT_KERNEL record fills the launch shape from snake_case attrs."""
    act = _FakeKernelActivity(
        name="k",
        start=0,
        end=1000,
        grid_x=8,
        grid_y=1,
        grid_z=1,
        block_x=128,
        block_y=2,
        block_z=1,
        static_shared_memory=512,
        dynamic_shared_memory=256,
        registers_per_thread=40,
    )
    trace = KernelTrace.from_activity(act)
    assert trace.grid == "8x1x1"
    assert trace.block == "128x2x1"
    assert trace.shared_mem == 768
    assert trace.duration_us == 1.0


def test_activity_record_duration() -> None:
    """A generic activity record reports its span."""
    assert ActivityRecord(start_ns=10, end_ns=60).duration_ns == 50


def test_bottleneck_report_splits_compute_and_copy() -> None:
    """The deep report splits GPU time into compute vs copy and ranks hot spots."""
    report = _traced_profile().trace_report()
    assert isinstance(report, BottleneckReport)
    assert report.compute_pct > report.memcpy_pct
    assert report.hot_kernels[0].name == "gemm"
    assert report.hot_regions[0].name in {"encode", "decode"}


def test_bottleneck_report_attributes_kernel_outside_any_window() -> None:
    """A kernel landing in no region window is labeled rather than dropped."""
    profile = Profile(
        windows=(RegionWindow(name="r", start_ns=0, end_ns=10, wall_ns=10),),
        kernels=(KernelTrace(name="stray", start_ns=100, end_ns=200),),
    )
    report = profile.trace_report()
    assert report.hot_regions[0].name == "(outside regions)"


def test_bottleneck_report_empty_profile() -> None:
    """No traces yields zero totals and empty rankings without dividing by zero."""
    report = Profile().trace_report()
    assert report.total_kernel_ns == 0
    assert report.hot_kernels == ()


def test_hot_region_uses_the_narrowest_window() -> None:
    profile = Profile(
        windows=(
            RegionWindow(name="inner", start_ns=100, end_ns=300, wall_ns=200),
            RegionWindow(name="outer", start_ns=0, end_ns=1000, wall_ns=1000),
        ),
        kernels=(KernelTrace(name="k", start_ns=150, end_ns=200),),
    )
    assert profile.trace_report().hot_regions[0].name == "inner"


def test_trace_collector_base_is_a_noop_context() -> None:
    """The base collector is an empty, safe no-op context manager."""
    with TraceCollector() as collector:
        collector.flush()
        collector.reset()
    assert collector.kernels() == []
    assert collector.memcpys() == []
    assert collector.activities() == []
    assert collector.dropped() == 0


def test_callback_session_base_counts_nothing() -> None:
    """The base callback session is a no-op that observes no calls."""
    with CallbackSession():
        pass


def test_kernel_trace_threads_per_block_ignores_nonnumeric() -> None:
    """A malformed block string degrades to the numeric dims it can parse."""
    assert KernelTrace(block="").threads_per_block == 1
    assert KernelTrace(block="16xNx2").threads_per_block == 32  # N -> 1


def test_kernel_trace_shared_mem_is_the_sum() -> None:
    """`shared_mem` stays the static+dynamic total for backward-compatible reads."""
    kernel = KernelTrace(static_shared_mem=100, dynamic_shared_mem=40)
    assert kernel.shared_mem == 140
