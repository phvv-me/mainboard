from mainboard import Diagnosis, Memory, Meter, Utilization
from mainboard.enums import Vendor
from mainboard.gpu import GPU
from mainboard.models.gpu_snapshot import GPUSnapshot
from mainboard.models.process_info import ProcessInfo
from mainboard.models.thermal_state import ThermalState
from mainboard.models.throttle_reason import ThrottleReason


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
    throttle: int = 0,
    processes: tuple[ProcessInfo, ...] = (),
) -> GPUSnapshot:
    """A synthetic snapshot exposing the utilization, thermal, and processes fields."""
    return GPUSnapshot(
        utilization=Utilization(gpu_pct=gpu_pct),
        thermal=ThermalState(throttle_reasons=throttle),
        processes=list(processes),
    )


def test_near_oom_fires_at_the_headroom_boundary() -> None:
    """Peak within 5% of capacity flags near-OOM and renders GB and percent."""
    healthy = Diagnosis.of(meter_with(peak_gpu_gb=80.0), snapshot(), capacity_gb=80.0)
    assert healthy.near_oom
    assert healthy.reason == "near OOM: 80.0/80.0 GB (100%)"

    edge = Diagnosis.of(meter_with(peak_gpu_gb=76.0), snapshot(), capacity_gb=80.0)
    assert edge.near_oom  # 95.0% sits exactly on the 100 - 5 headroom floor


def test_near_oom_is_off_with_comfortable_headroom() -> None:
    """Peak below the headroom floor leaves near-OOM clear."""
    clear = Diagnosis.of(meter_with(peak_gpu_gb=70.0), snapshot(), capacity_gb=80.0)
    assert not clear.near_oom
    assert clear.reason == "healthy"


def test_gpu_underutilized_at_and_below_the_floor() -> None:
    """Compute at or under the util floor flags underutilization with a percent reason."""
    idle = Diagnosis.of(meter_with(peak_gpu_gb=10.0), snapshot(gpu_pct=12), capacity_gb=80.0)
    assert idle.gpu_underutilized
    assert idle.reason == "GPU underutilized: 12% compute"

    busy = Diagnosis.of(meter_with(peak_gpu_gb=10.0), snapshot(gpu_pct=80), capacity_gb=80.0)
    assert not busy.gpu_underutilized


def test_throttled_reports_the_non_benign_reason() -> None:
    """A non-benign throttle reason fires and is named, outranking an idle GPU."""
    hot = Diagnosis.of(
        meter_with(peak_gpu_gb=10.0),
        snapshot(gpu_pct=10, throttle=int(ThrottleReason.SW_THERMAL_SLOWDOWN)),
        capacity_gb=80.0,
    )
    assert hot.throttled
    assert hot.reason == "throttled: SW_THERMAL_SLOWDOWN"


def test_benign_throttle_does_not_fire() -> None:
    """An idle-clock throttle is benign and leaves the throttled flag clear."""
    idle_clock = Diagnosis.of(
        meter_with(peak_gpu_gb=10.0),
        snapshot(gpu_pct=90, throttle=int(ThrottleReason.GPU_IDLE)),
        capacity_gb=80.0,
    )
    assert not idle_clock.throttled


def test_host_offload_from_memory_growth() -> None:
    """Sharp host memory growth flags an offload and renders the delta."""
    grew = Diagnosis.of(
        meter_with(peak_gpu_gb=10.0, host_delta_gb=6.0), snapshot(), capacity_gb=80.0
    )
    assert grew.host_offload
    assert grew.reason == "host offload: host memory grew 6.0 GB"


def test_host_offload_from_shared_processes() -> None:
    """An extra process sharing the GPU flags contention and is counted in the reason."""
    shared = Diagnosis.of(
        meter_with(peak_gpu_gb=10.0),
        snapshot(processes=(ProcessInfo(pid=1), ProcessInfo(pid=2))),
        capacity_gb=80.0,
    )
    assert shared.host_offload
    assert shared.reason == "host offload: 2 processes share the GPU"


def test_reason_orders_near_oom_above_every_other_flag() -> None:
    """With several flags at once near-OOM wins the one-line reason."""
    crowded = Diagnosis.of(
        meter_with(peak_gpu_gb=79.0, host_delta_gb=8.0),
        snapshot(gpu_pct=5, throttle=int(ThrottleReason.HW_SLOWDOWN)),
        capacity_gb=80.0,
    )
    assert crowded.near_oom and crowded.throttled and crowded.host_offload
    assert crowded.gpu_underutilized
    assert crowded.reason.startswith("near OOM:")


def test_zero_capacity_never_divides() -> None:
    """A device with no reported capacity reads as healthy rather than crashing."""
    unknown = Diagnosis.of(meter_with(peak_gpu_gb=10.0), snapshot(), capacity_gb=0.0)
    assert not unknown.near_oom
    assert unknown.reason == "healthy"


def test_diagnose_on_cpu_only_host_is_healthy() -> None:
    """With no GPU present `diagnose` returns the default healthy verdict."""
    out = Diagnosis.diagnose(meter_with(peak_gpu_gb=0.0), device_index=99)
    assert out == Diagnosis()
    assert out.reason == "healthy"


def test_diagnose_reads_the_live_device(monkeypatch) -> None:
    """`diagnose` snapshots the device and reads its capacity straight from the provider."""

    class TightGpu(GPU):
        vendor: Vendor = Vendor.NVIDIA

        @property
        def memory(self) -> Memory:
            return Memory(total_bytes=80 * 1024**3, used_bytes=79 * 1024**3)

        @property
        def utilization(self) -> Utilization:
            return Utilization(gpu_pct=95)

    monkeypatch.setattr(GPU, "all", classmethod(lambda cls: (TightGpu(index=0),)))
    out = Diagnosis.diagnose(meter_with(peak_gpu_gb=79.0))
    assert out.near_oom
    assert out.reason.startswith("near OOM:")
