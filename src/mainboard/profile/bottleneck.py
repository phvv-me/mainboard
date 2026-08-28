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
    """Run ``fn`` under the profiler and return its bottleneck report.

    fn: the zero-arg callable to profile (bind args with a lambda/partial).
    gpu: the device to profile and score bandwidth against. Left ``None`` the session
        discovers the host's own devices, so this is an override for picking one of
        several. Bandwidth is scored against whichever device the session ended up on,
        discovered or named, so a report never loses its peak for want of an argument.
    iters: timed runs of ``fn`` bracketed in one trace pass; warmup: untimed runs first.
    sync: a device barrier after each run (e.g. ``torch.cuda.synchronize``) so async GPU
        work is captured rather than just the launch.
    kinds: the :class:`Activity` kinds to request; adapted down to what the device
        supports, with the dropped kinds recorded in :attr:`ProfileReport.unavailable`.
    """
    supported = tracer().supported()
    granted = kinds & supported if supported else Activity(0)
    session = _run(fn, iters=iters, warmup=warmup, sync=sync, kinds=granted, gpu=gpu)
    scored = session.gpu
    return ProfileReport.from_profile(
        session.result(),
        iterations=iters,
        peak_bandwidth_gbps=scored.peak_bandwidth_gbs if scored is not None else 0.0,
        # only flag dropped kinds when a backend offered *some* tracing; with no
        # backend at all (CPU-only host) there is nothing to call unavailable.
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
    """Warm up, run `iters` timed passes inside one traced `span`, and hand back the session.

    The finished session rather than its result, since the caller also needs the device the
    session settled on, which is the discovered one whenever it was handed none.
    """
    for _ in range(warmup):
        fn()
    if sync is not None:
        sync()
    # ACTIVITY is requested only when a backend granted kinds to collect, since asking a
    # host with no tracing backend for a deep trace is a request nobody can serve and the
    # profiler now refuses it rather than returning an empty pass. With no `gpu` named the
    # profiler discovers the host's own, so a GPU host traces without a hand-carried probe.
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
    """Whether ``gpu`` is under load right now (someone else is using it).

    Busy means compute utilization above ``util_threshold`` percent or memory above
    ``memory_threshold_pct`` of capacity. Returns ``False`` when ``gpu`` is ``None``, so
    a CPU-only host always reads as idle.
    """
    if gpu is None:
        return False
    busy_compute = gpu.utilization.gpu_pct > util_threshold
    busy_memory = gpu.memory.percent_used > memory_threshold_pct
    return busy_compute or busy_memory


def wait_for_idle(
    gpu: BusyDevice | None,
    *,
    timeout: float = 30.0,
    poll_interval: float = 0.5,
    util_threshold: int = 10,
    memory_threshold_pct: float = 90.0,
    sleep: Callable[[float], None] = time.sleep,
) -> bool:
    """Block until ``gpu`` is idle, returning whether it became idle in ``timeout``.

    Polls :func:`gpu_busy` every ``poll_interval`` seconds. Returns ``True`` the moment the
    device is idle (immediately if it already is), or ``False`` once ``timeout`` seconds
    elapse while still busy — so a caller can decide to profile anyway or abort.
    sleep: the wait primitive, injected so tests need not spend real time.
    """
    deadline = time.monotonic() + timeout
    while gpu_busy(gpu, util_threshold=util_threshold, memory_threshold_pct=memory_threshold_pct):
        if time.monotonic() >= deadline:
            return False
        sleep(poll_interval)
    return True
