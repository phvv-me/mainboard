"""Turn a trial's raw hardware metrics into one actionable verdict.

:class:`Diagnosis` reads the signals an experiment already has after a
:func:`~mainboard.models.meter.meter` block, the :class:`~mainboard.models.meter.Meter`
and a :class:`~mainboard.models.gpu_snapshot.GPUSnapshot`, and answers the question an
operator asks when a run is slow or dies: what was the hardware doing. It raises four
flags, near-OOM (the early warning before an exit-137 SIGKILL), GPU underutilized
(CPU-bound or kernel-launch-bound), host offload (memory thrash or GPU contention), and
throttled (a non-benign NVML throttle reason), and renders the dominant one as a
human-readable line. It is a pure value object, inputs to verdict, with no I/O.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..gpu import GPU
from ..models.base import FrozenModel

if TYPE_CHECKING:
    from ..models.gpu_snapshot import GPUSnapshot
    from ..models.meter import Meter


class Diagnosis(FrozenModel):
    """One-line hardware verdict for a metered trial.

    near_oom: peak GPU memory came within ``headroom_pct`` of device capacity.
    gpu_underutilized: GPU compute utilization stayed below ``util_floor`` while running.
    host_offload: host memory grew sharply or extra processes shared the GPU.
    throttled: a non-benign NVML throttle reason (thermal or power) was active.
    reason: the dominant flag rendered human-readable, or ``healthy`` when none fired.
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
        *,
        device_index: int = 0,
        headroom_pct: float = 5.0,
        util_floor: int = 25,
        host_growth_gb: float = 4.0,
    ) -> Diagnosis:
        """Diagnose a finished trial against the live device, in one call.

        The framework path: snapshot the current GPU and read its capacity straight from
        the provider, so a caller hands over only the closed ``meter``. With no GPU
        present (a CPU-only host) every memory and utilization flag is off, so the verdict
        is ``healthy``.

        meter: the closed :class:`Meter` from the trial's ``with`` block.
        device_index: which GPU from :meth:`GPU.all` to read (default the first).
        """
        gpus = GPU.all()
        if device_index >= len(gpus):
            return cls()
        gpu = gpus[device_index]
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
        gpu: GPUSnapshot,
        *,
        capacity_gb: float,
        headroom_pct: float = 5.0,
        util_floor: int = 25,
        host_growth_gb: float = 4.0,
    ) -> Diagnosis:
        """Diagnose a finished trial from its meter and a final GPU snapshot.

        meter: the closed :class:`Meter`, read for ``peak_gpu_gb`` and ``host_delta_gb``.
        gpu: a :class:`GPUSnapshot` taken near the trial, for utilization, thermal, and
            shared processes.
        capacity_gb: total device memory in gibibytes, the denominator for near-OOM.
        headroom_pct: how close to capacity peak GPU memory must come to flag near-OOM.
        util_floor: compute utilization at or below this percent flags underutilization.
        host_growth_gb: host memory growth at or above this flags an offload.
        """
        used_pct = 100.0 * meter.peak_gpu_gb / capacity_gb if capacity_gb else 0.0
        near_oom = used_pct >= 100.0 - headroom_pct
        compute_pct = gpu.utilization.gpu_pct
        gpu_underutilized = compute_pct <= util_floor
        contended = len(gpu.processes) > 1
        host_offload = meter.host_delta_gb >= host_growth_gb or contended
        throttled = gpu.thermal.is_throttling
        return cls(
            near_oom=near_oom,
            gpu_underutilized=gpu_underutilized,
            host_offload=host_offload,
            throttled=throttled,
            reason=cls._reason(
                meter=meter,
                gpu=gpu,
                capacity_gb=capacity_gb,
                used_pct=used_pct,
                compute_pct=compute_pct,
                contended=contended,
                near_oom=near_oom,
                throttled=throttled,
                gpu_underutilized=gpu_underutilized,
                host_offload=host_offload,
            ),
        )

    @staticmethod
    def _reason(
        *,
        meter: Meter,
        gpu: GPUSnapshot,
        capacity_gb: float,
        used_pct: float,
        compute_pct: int,
        contended: bool,
        near_oom: bool,
        throttled: bool,
        gpu_underutilized: bool,
        host_offload: bool,
    ) -> str:
        """Render the dominant flag, near-OOM first as the most urgent, then a throttle.

        The order encodes severity: an imminent OOM kill outranks a throttle, which
        outranks contention or offload thrash, which outranks an idle GPU. With no flag
        the trial reads as ``healthy``.
        """
        if near_oom:
            return f"near OOM: {meter.peak_gpu_gb:.1f}/{capacity_gb:.1f} GB ({used_pct:.0f}%)"
        if throttled:
            return f"throttled: {', '.join(gpu.thermal.throttle_names)}"
        if host_offload:
            if contended:
                return f"host offload: {len(gpu.processes)} processes share the GPU"
            return f"host offload: host memory grew {meter.host_delta_gb:.1f} GB"
        if gpu_underutilized:
            return f"GPU underutilized: {compute_pct}% compute"
        return "healthy"
