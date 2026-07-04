"""One-call bottleneck profiling and GPU-contention gating.

Two helpers a kernel-optimization caller reaches for directly, so neither has to
hand-assemble a CUPTI collector again:

:func:`profile` runs a callable under the existing :class:`Profiler` and returns a
structured :class:`ProfileReport` — dominant kernel, memory-vs-compute verdict, occupancy
and shared-memory shape, and the per-kernel breakdown. It degrades per device: it asks
for the kinds the device actually supports, so an unsupported kind (e.g. ``MEMORY`` on
GB10) is simply absent from the report, never a crash.

:func:`gpu_busy` / :func:`wait_for_idle` read the live NVML utilization and memory the
provider already exposes, so a caller can wait for a clean measurement window before
profiling rather than timing against someone else's load.
"""

import time
from typing import TYPE_CHECKING

from ..gpu import GPU
from .annotate import region, tracer
from .profiler import Profiler
from .report import ProfileReport
from .trace import Activity

if TYPE_CHECKING:
    from collections.abc import Callable

    from .result import Profile


def profile[T, S](
    fn: Callable[[], T],
    *,
    iters: int = 1,
    warmup: int = 0,
    sync: Callable[[], S] | None = None,
    kinds: Activity = Activity.DEFAULT,
    device_index: int = 0,
) -> ProfileReport:
    """Run ``fn`` under the profiler and return its bottleneck report.

    fn: the zero-arg callable to profile (bind args with a lambda/partial).
    iters: timed runs of ``fn`` bracketed in one trace pass; warmup: untimed runs first.
    sync: a device barrier after each run (e.g. ``torch.cuda.synchronize``) so async GPU
        work is captured rather than just the launch.
    kinds: the :class:`Activity` kinds to request; adapted down to what the device
        supports, with the dropped kinds recorded in :attr:`ProfileReport.unavailable`.
    device_index: which GPU from ``GPU.all()`` to profile and score against its peak.
    """
    gpus = GPU.all()
    gpu = gpus[device_index] if device_index < len(gpus) else (gpus[0] if gpus else GPU())
    supported = tracer().supported()
    granted = kinds & supported if supported else Activity(0)
    profile_result = _run(fn, iters, warmup, sync, granted, device_index)
    return ProfileReport.from_profile(
        profile_result,
        iterations=iters,
        peak_bandwidth_gbps=gpu.peak_bandwidth_gbs,
        # only flag dropped kinds when a backend offered *some* tracing; with no
        # backend at all (CPU-only host) there is nothing to call unavailable.
        supported=supported.value if supported else None,
        requested=kinds.value if supported else None,
    )


def _run[T, S](
    fn: Callable[[], T],
    iters: int,
    warmup: int,
    sync: Callable[[], S] | None,
    kinds: Activity,
    device_index: int,
) -> Profile:
    """Warm up, then run ``iters`` timed passes inside one traced ``region``."""
    for _ in range(warmup):
        fn()
    if sync is not None:
        sync()
    with Profiler(trace=kinds, device_index=device_index) as profiler:
        for _ in range(iters):
            with region("fn"):
                fn()
            if sync is not None:
                sync()
    return profiler.result()


def gpu_busy(
    index: int = 0,
    *,
    util_threshold: int = 10,
    memory_threshold_pct: float = 90.0,
) -> bool:
    """Whether GPU ``index`` is under load right now (someone else is using it).

    Busy means compute utilization above ``util_threshold`` percent or memory above
    ``memory_threshold_pct`` of capacity. Returns ``False`` when no GPU is present, so a
    CPU-only host always reads as idle.
    """
    gpus = GPU.all()
    if index >= len(gpus):
        return False
    gpu = gpus[index]
    busy_compute = gpu.utilization.gpu_pct > util_threshold
    busy_memory = gpu.memory.percent_used > memory_threshold_pct
    return busy_compute or busy_memory


def wait_for_idle(
    index: int = 0,
    *,
    timeout: float = 30.0,
    poll_interval: float = 0.5,
    util_threshold: int = 10,
    memory_threshold_pct: float = 90.0,
    sleep: Callable[[float], None] = time.sleep,
) -> bool:
    """Block until GPU ``index`` is idle, returning whether it became idle in ``timeout``.

    Polls :func:`gpu_busy` every ``poll_interval`` seconds. Returns ``True`` the moment the
    device is idle (immediately if it already is), or ``False`` once ``timeout`` seconds
    elapse while still busy — so a caller can decide to profile anyway or abort.
    sleep: the wait primitive, injected so tests need not spend real time.
    """
    deadline = time.monotonic() + timeout
    while gpu_busy(
        index, util_threshold=util_threshold, memory_threshold_pct=memory_threshold_pct
    ):
        if time.monotonic() >= deadline:
            return False
        sleep(poll_interval)
    return True
