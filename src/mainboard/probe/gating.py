import time

from ..profile.bottleneck import gpu_busy as device_busy
from .machine import Machine


def gpu_busy(
    *,
    util_threshold: int = 10,
    memory_threshold_pct: float = 90.0,
) -> bool:
    """Return whether the first visible GPU is currently busy."""
    gpus = Machine().gpus
    return device_busy(
        gpus[0] if gpus else None,
        util_threshold=util_threshold,
        memory_threshold_pct=memory_threshold_pct,
    )


def wait_for_idle(
    *,
    timeout: float = 30.0,
    idle_duration: float = 0.0,
    poll_interval: float = 0.5,
    util_threshold: int = 10,
    memory_threshold_pct: float = 90.0,
) -> bool:
    """Wait until the first visible GPU remains idle for the requested duration."""
    deadline = time.monotonic() + timeout
    idle_since: float | None = None
    while True:
        now = time.monotonic()
        if not gpu_busy(
            util_threshold=util_threshold,
            memory_threshold_pct=memory_threshold_pct,
        ):
            idle_since = now if idle_since is None else idle_since
            if now - idle_since >= idle_duration:
                return True
        else:
            idle_since = None
        if now >= deadline:
            return False
        time.sleep(min(poll_interval, max(deadline - now, 0.0)))
