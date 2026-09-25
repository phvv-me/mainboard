from patos import FrozenModel

from .meter import Meter
from .protocols import DeviceProbe, DeviceSnapshot


class Diagnosis(FrozenModel):
    """One-line hardware verdict for a metered trial.

    near_oom: peak GPU memory came within `headroom_pct` of device capacity.
    gpu_underutilized: compute utilization at or below `util_floor` percent.
    host_offload: host memory grew by `host_growth_gb` or more, or other processes shared the GPU.
    throttled: a non-benign thermal/power throttle was active.
    reason: the most severe flag rendered, or `healthy` when none fired.
    """

    near_oom: bool = False
    gpu_underutilized: bool = False
    host_offload: bool = False
    throttled: bool = False
    reason: str = "healthy"

    @classmethod
    def diagnose(
        cls,
        meter: Meter,
        gpu: DeviceProbe | None,
        *,
        headroom_pct: float = 5.0,
        util_floor: int = 25,
        host_growth_gb: float = 4.0,
    ) -> Diagnosis:
        """Diagnose a closed `meter` against a snapshot of the live `gpu` and its capacity.

        gpu: None on a CPU-only host, which reads healthy (mainboard.probe is not a dependency
            of profiling, so the caller resolves the device).
        """
        if gpu is None:
            return cls()
        return cls.of(
            meter,
            gpu.snapshot(),
            capacity_gb=gpu.memory.total_gb,
            headroom_pct=headroom_pct,
            util_floor=util_floor,
            host_growth_gb=host_growth_gb,
        )

    @classmethod
    def of(
        cls,
        meter: Meter,
        gpu: DeviceSnapshot,
        *,
        capacity_gb: float,
        headroom_pct: float = 5.0,
        util_floor: int = 25,
        host_growth_gb: float = 4.0,
    ) -> Diagnosis:
        """Diagnose a closed `meter` from a device snapshot taken near the trial.

        capacity_gb: total device memory, the near-OOM denominator (0 reads as none used).
        """
        used_pct = 100.0 * meter.peak_gpu_gb / capacity_gb if capacity_gb else 0.0
        near_oom = used_pct >= 100.0 - headroom_pct
        compute_pct = gpu.utilization.gpu_pct
        gpu_underutilized = compute_pct <= util_floor
        contended = len(gpu.processes) > 1
        host_offload = meter.host_delta_gb >= host_growth_gb or contended
        throttled = gpu.thermal.is_throttling
        # Severity order: an imminent OOM kill outranks a throttle, which outranks contention
        # or offload thrash, which outranks an idle GPU.
        renderings = [
            (
                near_oom,
                f"near OOM: {meter.peak_gpu_gb:.1f}/{capacity_gb:.1f} GB ({used_pct:.0f}%)",
            ),
            (throttled, f"throttled: {', '.join(gpu.thermal.throttle_names)}"),
            (
                host_offload and contended,
                f"host offload: {len(gpu.processes)} processes share the GPU",
            ),
            (host_offload, f"host offload: host memory grew {meter.host_delta_gb:.1f} GB"),
            (gpu_underutilized, f"GPU underutilized: {compute_pct}% compute"),
        ]
        return cls(
            near_oom=near_oom,
            gpu_underutilized=gpu_underutilized,
            host_offload=host_offload,
            throttled=throttled,
            reason=next((told for fired, told in renderings if fired), "healthy"),
        )
