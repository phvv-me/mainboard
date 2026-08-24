import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from mainboard.profile import (
    Activity,
    Bound,
    MemcpyTrace,
    Profile,
    ProfileReport,
    RegionSummary,
)

from .conftest import kernel


def _report(profile: Profile, *, peak_bandwidth_gbps: float = 0.0) -> ProfileReport:
    """The bottleneck verdict for one profile, over a single iteration."""
    return ProfileReport.from_profile(
        profile, iterations=1, peak_bandwidth_gbps=peak_bandwidth_gbps
    )


def _sampled(*, util_pct: float, memory_util_pct: float) -> Profile:
    """A profile carrying one sampled region's utilization and nothing traced."""
    return Profile(
        summaries=(
            RegionSummary(
                name="r",
                wall_ms=1.0,
                avg_util_pct=util_pct,
                avg_memory_util_pct=memory_util_pct,
            ),
        )
    )


@pytest.mark.parametrize(
    ("profile", "bound"),
    [
        (
            Profile(kernels=(kernel("gemm", 1000),), memcpys=(MemcpyTrace(end_ns=100),)),
            Bound.COMPUTE,
        ),
        (
            Profile(kernels=(kernel("k", 100),), memcpys=(MemcpyTrace(end_ns=1000),)),
            Bound.MEMORY,
        ),
        (_sampled(util_pct=5.0, memory_util_pct=80.0), Bound.MEMORY),
        (_sampled(util_pct=90.0, memory_util_pct=3.0), Bound.COMPUTE),
        (Profile(), Bound.UNKNOWN),
    ],
    ids=[
        "kernels_dominate_the_gpu_time",
        "copies_dominate_the_gpu_time",
        "no_traces_so_the_memory_controller_decides",
        "no_traces_so_the_sms_decide",
        "nothing_to_classify",
    ],
)
def test_the_bound_verdict_follows_the_time_split_then_the_utilization_signal(
    profile: Profile, bound: Bound
) -> None:
    """Copies out-timing kernels reads memory-bound, and with no time at all util decides.

    With neither a traced duration nor a sampled utilization there is nothing to judge, so
    the verdict is UNKNOWN and the report says no kernels were traced.
    """
    report = _report(profile)
    assert report.bound is bound
    if bound is Bound.UNKNOWN:
        assert report.dominant_kernel == ""
        assert "No kernels traced" in report.report()


# A short list of durations is a small space, so a trimmed budget covers it and keeps the
# suite's wall time where it was.
@settings(max_examples=15)
@given(durations=st.lists(st.integers(min_value=1, max_value=10_000), min_size=1, max_size=8))
@example(durations=[100, 100, 500])  # two calls of one name against a hotter single call
def test_kernel_shares_partition_the_kernel_time_hottest_first(durations: list[int]) -> None:
    """Per-kernel shares always add back up to 100% and rank the hottest name first.

    Repeated launches of one name collapse into a single row carrying the call count and
    the summed time, which is what makes the dominant kernel the dominant *name*.
    """
    kernels = tuple(kernel(f"k{index % 2}", ns) for index, ns in enumerate(durations))
    report = _report(Profile(kernels=kernels))
    totals = [stat.total_ns for stat in report.kernels]

    assert abs(sum(stat.share_pct for stat in report.kernels) - 100.0) < 1e-6
    assert totals == sorted(totals, reverse=True)
    assert sum(stat.calls for stat in report.kernels) == len(durations)
    assert sum(totals) == sum(durations)
    assert report.dominant_kernel == report.kernels[0].name
    assert report.dominant_share_pct == report.kernels[0].share_pct
    assert all(stat.avg_ns == stat.total_ns / stat.calls for stat in report.kernels)


def test_kernel_stat_carries_launch_shape() -> None:
    """Occupancy proxy, registers, and the static/dynamic shared split are reported."""
    profile = Profile(
        kernels=(
            kernel(
                "k",
                1000,
                block="512x1x1",
                registers=48,
                static_shared_mem=2048,
                dynamic_shared_mem=1024,
            ),
        )
    )
    stat = _report(profile).kernels[0]
    assert stat.threads_per_block == 512
    assert stat.occupancy_pct == 50.0  # 512 / 1024
    assert stat.registers == 48
    assert stat.static_shared_mem == 2048
    assert stat.dynamic_shared_mem == 1024


@pytest.mark.parametrize(
    ("end_ns", "achieved_gbps"),
    [(1000, 2.0), (0, 0.0)],
    ids=["bytes_over_copy_time", "a_copy_with_no_measured_time"],
)
def test_copy_bandwidth_is_bytes_over_copy_time_scored_against_peak(
    end_ns: int, achieved_gbps: float
) -> None:
    """Achieved copy bandwidth is shown against the device peak, never divided by zero."""
    profile = Profile(memcpys=(MemcpyTrace(end_ns=end_ns, bytes_moved=2000),))
    report = _report(profile, peak_bandwidth_gbps=10.0)
    assert report.achieved_bandwidth_gbps == achieved_gbps
    assert "of peak" in report.report()


def test_sampled_memory_becomes_the_high_water_mark_and_the_mean() -> None:
    """The report surfaces the largest sampled footprint, and says nothing when none was taken.

    A kernel that finished between two sampler ticks reports zero rather than a note, since
    the deep kernel trace stays the reliable signal there.
    """
    profile = Profile(
        summaries=(
            RegionSummary(name="r", wall_ms=1.0, peak_memory_bytes=300, avg_memory_bytes=200),
            RegionSummary(name="r", wall_ms=1.0, peak_memory_bytes=900, avg_memory_bytes=600),
        )
    )
    sampled = _report(profile)
    assert sampled.peak_memory_bytes == 900  # max single sample, the footprint to fit
    assert sampled.avg_memory_bytes == 400  # (200 + 600) / 2
    assert "peak memory" in sampled.report()

    unsampled = _report(Profile(kernels=(kernel("k", 100),)))
    assert unsampled.peak_memory_bytes == 0
    assert unsampled.avg_memory_bytes == 0
    assert "peak memory" not in unsampled.report()


def test_the_rendered_report_names_the_device_and_any_untraced_kinds() -> None:
    """`str` is the plain-text verdict, and kinds the device could not trace are listed.

    Without a support/request pair there is nothing to mark unavailable, and an empty
    device name renders as `cpu` rather than as a blank.
    """
    named = _report(Profile(device="dev", kernels=(kernel("k", 100),)))
    assert str(named) == named.report()
    assert "dev" in str(named)
    assert named.unavailable == ()

    partial = ProfileReport.from_profile(
        Profile(),
        iterations=1,
        peak_bandwidth_gbps=0.0,
        supported=Activity.KERNEL.value,
        requested=Activity.ALL.value,
    )
    assert "memcpy" in partial.unavailable
    assert "kernel" not in partial.unavailable
    assert "all" not in partial.unavailable
    assert "unavailable on this device" in partial.report()
    assert "device cpu" in partial.report()
