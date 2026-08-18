import pytest
from hypothesis import given
from hypothesis import strategies as st

from mainboard.profile import (
    Activity,
    Bound,
    KernelStat,
    KernelTrace,
    MemcpyTrace,
    Profile,
    ProfileReport,
    RegionSummary,
)


def _kernel(name: str, ns: int, **shape: str | int) -> KernelTrace:
    """A `KernelTrace` of ``name`` lasting ``ns`` nanoseconds from t=0."""
    return KernelTrace(name=name, start_ns=0, end_ns=ns, **shape)  # pyrefly: ignore  reason=dynamic **shape kwargs lose field-precise typing since=2026-08-16


def test_classifies_compute_bound_when_kernels_dominate() -> None:
    """More kernel time than copy time reads as compute-bound."""
    profile = Profile(kernels=(_kernel("gemm", 1000),), memcpys=(MemcpyTrace(end_ns=100),))
    report = ProfileReport.from_profile(profile, iterations=1, peak_bandwidth_gbps=0.0)
    assert report.bound is Bound.COMPUTE


def test_classifies_memory_bound_when_copies_dominate() -> None:
    """More copy time than kernel time reads as memory-bound."""
    profile = Profile(
        kernels=(_kernel("k", 100),),
        memcpys=(MemcpyTrace(end_ns=1000, bytes_moved=4096),),
    )
    report = ProfileReport.from_profile(profile, iterations=1, peak_bandwidth_gbps=10.0)
    assert report.bound is Bound.MEMORY


def test_falls_back_to_utilization_when_no_traces() -> None:
    """With no kernel/copy time, the sampled util decides the verdict."""

    busy_memory = Profile(
        summaries=(
            RegionSummary(name="r", wall_ms=1.0, avg_util_pct=5.0, avg_memory_util_pct=80.0),
        )
    )
    busy_compute = Profile(
        summaries=(
            RegionSummary(name="r", wall_ms=1.0, avg_util_pct=90.0, avg_memory_util_pct=3.0),
        )
    )
    assert ProfileReport.from_profile(busy_memory, iterations=1, peak_bandwidth_gbps=0).bound is (
        Bound.MEMORY
    )
    assert ProfileReport.from_profile(busy_compute, iterations=1, peak_bandwidth_gbps=0).bound is (
        Bound.COMPUTE
    )


def test_unknown_when_nothing_to_classify() -> None:
    """An empty profile cannot be classified, so the verdict is UNKNOWN."""
    report = ProfileReport.from_profile(Profile(), iterations=1, peak_bandwidth_gbps=0.0)
    assert report.bound is Bound.UNKNOWN
    assert report.dominant_kernel == ""
    assert "No kernels traced" in report.report()


def test_dominant_kernel_is_the_hottest_by_total_time() -> None:
    """The breakdown ranks kernels by total time and names the hottest dominant."""
    profile = Profile(
        kernels=(_kernel("a", 100), _kernel("a", 100), _kernel("b", 500)),
    )
    report = ProfileReport.from_profile(profile, iterations=1, peak_bandwidth_gbps=0.0)
    assert report.dominant_kernel == "b"
    assert [k.name for k in report.kernels] == ["b", "a"]
    kernel_a = next(k for k in report.kernels if k.name == "a")
    assert kernel_a.calls == 2
    assert kernel_a.total_ns == 200


def test_kernel_stat_carries_launch_shape() -> None:
    """Occupancy proxy, registers, and the static/dynamic shared split are reported."""
    profile = Profile(
        kernels=(
            _kernel(
                "k",
                1000,
                block="512x1x1",
                registers=48,
                static_shared_mem=2048,
                dynamic_shared_mem=1024,
            ),
        )
    )
    stat = ProfileReport.from_profile(profile, iterations=1, peak_bandwidth_gbps=0.0).kernels[0]
    assert stat.threads_per_block == 512
    assert stat.occupancy_pct == 50.0  # 512 / 1024
    assert stat.registers == 48
    assert stat.static_shared_mem == 2048
    assert stat.dynamic_shared_mem == 1024


def test_copy_bandwidth_scored_against_peak() -> None:
    """Achieved copy bandwidth is bytes over copy-time, shown against the device peak."""
    profile = Profile(memcpys=(MemcpyTrace(end_ns=1000, bytes_moved=2000),))
    report = ProfileReport.from_profile(profile, iterations=1, peak_bandwidth_gbps=10.0)
    assert report.achieved_bandwidth_gbps == 2.0  # 2000 bytes / 1000 ns
    assert "of peak" in report.report()


def test_zero_duration_copy_has_zero_bandwidth() -> None:
    """A copy with no measured time yields no bandwidth rather than dividing by zero."""
    profile = Profile(memcpys=(MemcpyTrace(end_ns=0, bytes_moved=2000),))
    report = ProfileReport.from_profile(profile, iterations=1, peak_bandwidth_gbps=10.0)
    assert report.achieved_bandwidth_gbps == 0.0


def test_peak_memory_is_the_high_water_mark_across_regions() -> None:
    """The report surfaces the largest sampled memory footprint and the mean."""

    profile = Profile(
        summaries=(
            RegionSummary(name="r", wall_ms=1.0, peak_memory_bytes=300, avg_memory_bytes=200),
            RegionSummary(name="r", wall_ms=1.0, peak_memory_bytes=900, avg_memory_bytes=600),
        )
    )
    report = ProfileReport.from_profile(profile, iterations=1, peak_bandwidth_gbps=0.0)
    assert report.peak_memory_bytes == 900  # max single sample, the footprint to fit
    assert report.avg_memory_bytes == 400  # (200 + 600) / 2
    assert "peak memory" in report.report()


def test_peak_memory_absent_when_nothing_sampled() -> None:
    """A kernel that finished between sampler ticks reports zero peak memory, not a note."""
    report = ProfileReport.from_profile(
        Profile(kernels=(_kernel("k", 100),)), iterations=1, peak_bandwidth_gbps=0.0
    )
    assert report.peak_memory_bytes == 0
    assert report.avg_memory_bytes == 0
    assert "peak memory" not in report.report()


def test_unavailable_lists_dropped_activity_kinds() -> None:
    """Kinds requested but unsupported by the device surface in `unavailable`."""
    report = ProfileReport.from_profile(
        Profile(),
        iterations=1,
        peak_bandwidth_gbps=0.0,
        supported=Activity.KERNEL.value,
        requested=Activity.ALL.value,
    )
    assert "memcpy" in report.unavailable
    assert "kernel" not in report.unavailable
    assert "all" not in report.unavailable
    assert "unavailable on this device" in report.report()


def test_unavailable_empty_when_support_unknown() -> None:
    """Without a support/request pair there is nothing to mark unavailable."""
    report = ProfileReport.from_profile(Profile(), iterations=1, peak_bandwidth_gbps=0.0)
    assert report.unavailable == ()


def test_str_renders_the_report() -> None:
    """`str(report)` is the plain-text verdict table."""
    profile = Profile(device="dev", kernels=(_kernel("k", 100),))
    report = ProfileReport.from_profile(profile, iterations=1, peak_bandwidth_gbps=0.0)
    assert str(report) == report.report()
    assert "dev" in str(report)


def test_report_names_cpu_when_no_device() -> None:
    """An empty device name renders as `cpu` in the header."""
    report = ProfileReport.from_profile(Profile(), iterations=1, peak_bandwidth_gbps=0.0)
    assert "device cpu" in report.report()


@given(
    durations=st.lists(st.integers(min_value=1, max_value=10_000), min_size=1, max_size=8),
)
def test_kernel_shares_sum_to_one_hundred(durations: list[int]) -> None:
    """Per-kernel shares always partition 100% of kernel time (no rounding drift away)."""
    kernels = tuple(_kernel(f"k{i}", ns) for i, ns in enumerate(durations))
    report = ProfileReport.from_profile(
        Profile(kernels=kernels), iterations=1, peak_bandwidth_gbps=0.0
    )
    assert abs(sum(k.share_pct for k in report.kernels) - 100.0) < 1e-6


def test_kernel_stat_is_frozen() -> None:
    """The report models are immutable like the rest of the profiling schemas."""
    stat = KernelStat(
        name="k",
        calls=1,
        total_ns=1,
        avg_ns=1.0,
        share_pct=100.0,
        grid="",
        block="",
        threads_per_block=1,
        occupancy_pct=0.0,
        registers=0,
        static_shared_mem=0,
        dynamic_shared_mem=0,
    )
    with pytest.raises(ValueError, match="frozen"):
        stat.calls = 2  # pyrefly: ignore  reason=intentionally assigns a frozen field to prove it raises since=2026-08-16
