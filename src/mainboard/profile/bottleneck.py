# One-call bottleneck profiling and GPU-contention gating.

import time
from typing import TYPE_CHECKING

from .annotate import tracer
from .profiler import Profiler
from .report import ProfileReport
from .spans import span
from .trace import Activity

if TYPE_CHECKING:
    from collections.abc import Callable

    from .protocols import BusyDevice, DeviceProbe


def profile[T, S](
    fn: Callable[[], T],
    *,
    gpu: DeviceProbe | None = None,
    iters: int = 1,
    warmup: int = 0,
    sync: Callable[[], S] | None = None,
    kinds: Activity = Activity.DEFAULT,
) -> ProfileReport:
    """Run zero-arg `fn` `iters` times in one traced pass, after `warmup` untimed runs.

    gpu: overrides the session's own device discovery. Bandwidth is scored against whichever
        device the session settled on, so a report never loses its peak for want of this.
    sync: a device barrier after each run (e.g. `torch.cuda.synchronize`) so async GPU work
        is captured rather than just the launch.
    kinds: adapted down to what the device supports; the dropped kinds are recorded in
        `ProfileReport.unavailable`.
    """
    supported = tracer().supported()
    granted = kinds & supported if supported else Activity(0)
    session = _run(fn, iters=iters, warmup=warmup, sync=sync, kinds=granted, gpu=gpu)
    scored = session.gpu
    return ProfileReport.from_profile(
        session.result(),
        iterations=iters,
        peak_bandwidth_gbps=scored.peak_bandwidth_gbs if scored is not None else 0.0,
        # With no backend at all (a CPU-only host) nothing counts as unavailable.
        supported=supported.value if supported else None,
        requested=kinds.value if supported else None,
    )


def _run[T, S](
    fn: Callable[[], T],
    *,
    iters: int,
    warmup: int,
    sync: Callable[[], S] | None,
    kinds: Activity,
    gpu: DeviceProbe | None,
) -> Profiler:
    """The finished session rather than its result, since the caller also reads its device."""
    for _ in range(warmup):
        fn()
    if sync is not None:
        sync()
    # ACTIVITY only when a backend granted kinds, since the profiler refuses a deep trace
    # nobody can serve rather than returning an empty pass.
    deep = Profiler.Feature.ACTIVITY if kinds else Profiler.Feature(0)
    with Profiler(
        gpus=(gpu,) if gpu is not None else (),
        features=(
            Profiler.Feature.SPANS | Profiler.Feature.DEVICE | Profiler.Feature.MARKERS | deep
        ),
        activities=kinds,
    ) as profiler:
        for _ in range(iters):
            with span("fn"):
                fn()
            if sync is not None:
                sync()
    return profiler


def gpu_busy(
    gpu: BusyDevice | None,
    *,
    util_threshold: int = 10,
    memory_threshold_pct: float = 90.0,
) -> bool:
    """Whether `gpu` is under load, its compute or memory percent over the threshold.

    A CPU-only host (None) always reads as idle.
    """
    if gpu is None:
        return False
    return (
        gpu.utilization.gpu_pct > util_threshold or gpu.memory.percent_used > memory_threshold_pct
    )


def wait_for_idle(
    gpu: BusyDevice | None,
    *,
    timeout: float = 30.0,
    poll_interval: float = 0.5,
    util_threshold: int = 10,
    memory_threshold_pct: float = 90.0,
    sleep: Callable[[float], None] = time.sleep,
) -> bool:
    """Poll `gpu_busy` every `poll_interval` seconds; False if still busy after `timeout`.

    sleep: the wait primitive, injected so tests need not spend real time.
    """
    deadline = time.monotonic() + timeout
    while gpu_busy(gpu, util_threshold=util_threshold, memory_threshold_pct=memory_threshold_pct):
        if time.monotonic() >= deadline:
            return False
        sleep(poll_interval)
    return True
