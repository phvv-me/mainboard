import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from mainboard import Meter
from mainboard.profile import Diagnosis

from .conftest import FakeGPU, FakeMemory, FakeProcess, FakeSnapshot, FakeThermal, FakeUtilization


def meter_with(*, peak_gpu_gb: float = 0.0, host_delta_gb: float = 0.0) -> Meter:
    """A closed-style meter with the two readings the diagnosis reads, set directly."""
    gauge = Meter.__new__(Meter)
    gauge.host_used_gb = [0.0, host_delta_gb]
    gauge.gpu_used_gb = [peak_gpu_gb]
    gauge.elapsed_s = 1.0
    return gauge


def snapshot(
    *,
    gpu_pct: int = 90,
    is_throttling: bool = False,
    throttle_names: tuple[str, ...] = (),
    processes: tuple[FakeProcess, ...] = (),
) -> FakeSnapshot:
    """A synthetic snapshot exposing the utilization, thermal, and processes fields."""
    return FakeSnapshot(
        utilization=FakeUtilization(gpu_pct=gpu_pct),
        thermal=FakeThermal(is_throttling=is_throttling, throttle_names=throttle_names),
        processes=processes,
    )


_SHARED = (FakeProcess(pid=1, used_bytes=0), FakeProcess(pid=2, used_bytes=0))


@pytest.mark.parametrize(
    ("peak_gpu_gb", "host_delta_gb", "reading", "capacity_gb", "flags", "reason"),
    [
        (80.0, 0.0, snapshot(), 80.0, "near_oom", "near OOM: 80.0/80.0 GB (100%)"),
        (76.0, 0.0, snapshot(), 80.0, "near_oom", "near OOM: 76.0/80.0 GB (95%)"),
        (70.0, 0.0, snapshot(), 80.0, "", "healthy"),
        (
            10.0,
            0.0,
            snapshot(gpu_pct=12),
            80.0,
            "gpu_underutilized",
            "GPU underutilized: 12% compute",
        ),
        (
            10.0,
            0.0,
            snapshot(gpu_pct=10, is_throttling=True, throttle_names=("SW_THERMAL_SLOWDOWN",)),
            80.0,
            "gpu_underutilized throttled",
            "throttled: SW_THERMAL_SLOWDOWN",
        ),
        (10.0, 0.0, snapshot(gpu_pct=90, is_throttling=False), 80.0, "", "healthy"),
        (10.0, 6.0, snapshot(), 80.0, "host_offload", "host offload: host memory grew 6.0 GB"),
        (
            10.0,
            0.0,
            snapshot(processes=_SHARED),
            80.0,
            "host_offload",
            "host offload: 2 processes share the GPU",
        ),
        (
            79.0,
            8.0,
            snapshot(gpu_pct=5, is_throttling=True, throttle_names=("HW_SLOWDOWN",)),
            80.0,
            "near_oom gpu_underutilized host_offload throttled",
            "near OOM: 79.0/80.0 GB (99%)",
        ),
        (10.0, 0.0, snapshot(), 0.0, "", "healthy"),
    ],
    ids=[
        "at_capacity",
        "on_the_headroom_floor",
        "comfortable_headroom",
        "idle_gpu",
        "a_named_throttle_outranks_an_idle_gpu",
        "a_benign_idle_clock_throttle_does_not_fire",
        "host_memory_grew",
        "another_process_shares_the_device",
        "near_oom_outranks_every_other_flag",
        "a_device_that_reports_no_capacity",
    ],
)
def test_the_dominant_flag_decides_the_one_line_reason(
    peak_gpu_gb: float,
    host_delta_gb: float,
    reading: FakeSnapshot,
    capacity_gb: float,
    flags: str,
    reason: str,
) -> None:
    """Severity orders the verdict, near-OOM first, then a throttle, then thrash, then idle.

    A device that reports no capacity never divides by it, so it reads as healthy rather
    than crashing, and a throttle that never fired leaves the flag clear.
    """
    verdict = Diagnosis.of(
        meter_with(peak_gpu_gb=peak_gpu_gb, host_delta_gb=host_delta_gb),
        reading,
        capacity_gb=capacity_gb,
    )
    assert {name for name in Diagnosis.model_fields if getattr(verdict, name) is True} == set(
        flags.split()
    )
    assert verdict.reason == reason


# Four bounded numeric axes are a small space, so a trimmed budget covers them and keeps the
# suite's wall time where it was.
@settings(max_examples=15)
@given(
    peak_gpu_gb=st.floats(min_value=0.0, max_value=80.0, allow_nan=False, allow_infinity=False),
    extra_gb=st.floats(min_value=0.0, max_value=80.0, allow_nan=False, allow_infinity=False),
    gpu_pct=st.integers(min_value=0, max_value=100),
    rise_pct=st.integers(min_value=0, max_value=100),
)
def test_the_flags_move_only_one_way_with_the_readings_that_drive_them(
    peak_gpu_gb: float, extra_gb: float, gpu_pct: int, rise_pct: int
) -> None:
    """More memory never clears near-OOM, and more compute never flags an idle GPU.

    The verdict reads `healthy` exactly when no flag fired, so the one line and the flags
    can never disagree about whether the trial was clean.
    """
    reading = snapshot(gpu_pct=gpu_pct)
    verdict = Diagnosis.of(meter_with(peak_gpu_gb=peak_gpu_gb), reading, capacity_gb=80.0)
    fuller = Diagnosis.of(
        meter_with(peak_gpu_gb=peak_gpu_gb + extra_gb), reading, capacity_gb=80.0
    )
    busier = Diagnosis.of(
        meter_with(peak_gpu_gb=peak_gpu_gb),
        snapshot(gpu_pct=min(gpu_pct + rise_pct, 100)),
        capacity_gb=80.0,
    )
    assert fuller.near_oom >= verdict.near_oom
    assert busier.gpu_underutilized <= verdict.gpu_underutilized
    fired = verdict.near_oom or verdict.gpu_underutilized or verdict.host_offload
    assert (verdict.reason == "healthy") is not (fired or verdict.throttled)


def test_diagnose_reads_capacity_off_the_live_device_or_calls_a_cpu_host_healthy() -> None:
    """`diagnose` snapshots the device itself, and with no device every flag stays off."""
    gpu = FakeGPU(memory=FakeMemory(total_gb=80.0), reading=snapshot(gpu_pct=95))
    live = Diagnosis.diagnose(meter_with(peak_gpu_gb=79.0), gpu)
    assert live.near_oom
    assert live.reason.startswith("near OOM:")

    bare = Diagnosis.diagnose(meter_with(peak_gpu_gb=0.0), None)
    assert bare == Diagnosis()
    assert bare.reason == "healthy"
