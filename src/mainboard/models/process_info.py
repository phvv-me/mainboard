from .base import FrozenModel


class ProcessInfo(FrozenModel):
    """GPU process memory usage.

    pid: operating system process ID.
    used_bytes: GPU memory allocated by this process in bytes.
    """

    pid: int = 0
    used_bytes: int = 0
