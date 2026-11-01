import resource
from collections.abc import Generator
from contextlib import contextmanager, suppress
from threading import RLock

# Soft limits asked for, largest first, where the hard limit is unbounded: macOS refuses an
# infinite soft limit and grants up to its per-process ceiling (`kern.maxfilesperproc`, 92160 on
# Apple silicon), and 10240 (`OPEN_MAX`) always. Its default of 256 is below one lake table.
_UNBOUNDED = (1 << 20, 92160, 10240)


class FileBudget:
    """Share the permitted file limit across overlapping operations, restoring it last."""

    _lock = RLock()
    _holders = 0
    _original: tuple[int, int] = (0, 0)

    @classmethod
    @contextmanager
    def permitted(cls) -> Generator[None]:
        with cls._lock:
            if cls._holders == 0:
                cls._original = resource.getrlimit(resource.RLIMIT_NOFILE)
                FileBudget._raise(cls._original[1])
            cls._holders += 1
        try:
            yield
        finally:
            cls._release()

    @classmethod
    def _release(cls) -> None:
        """Give up one hold, restoring the original limit when it was the last."""
        with cls._lock:
            cls._holders -= 1
            if cls._holders == 0:
                resource.setrlimit(resource.RLIMIT_NOFILE, cls._original)

    @staticmethod
    def _raise(hard: int) -> None:
        """Lift the soft file limit to `hard`, or to the kernel's largest where that is unbounded.

        The current limit stays when the kernel grants none of them.
        """
        asked = (hard,) if hard != resource.RLIM_INFINITY else _UNBOUNDED
        for soft in asked:
            with suppress(ValueError, OSError):
                resource.setrlimit(resource.RLIMIT_NOFILE, (soft, hard))
                return
