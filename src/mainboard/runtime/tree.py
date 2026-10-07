# A command and everything it started, ended together.
#
# Signalling only the process a job spawned leaves its workers (a dataloader, a compile pool, a
# serving engine) running and holding the card. The tree is read from the process table rather
# than a process group, so the same code ends a job on Linux and macOS, and the command stays in
# the runner's group so a scheduler signalling the group (PBS, pueue) still reaches it.

import resource
import subprocess
from collections.abc import Generator
from contextlib import contextmanager, suppress
from threading import RLock

import psutil

# Soft limits asked for, largest first, where the hard limit is unbounded: macOS refuses an
# infinite soft limit and grants up to its per-process ceiling (`kern.maxfilesperproc`, 92160 on
# Apple silicon), and 10240 (`OPEN_MAX`) always. Its default of 256 is below one lake table.
_UNBOUNDED = (1 << 20, 92160, 10240)


def _raise(hard: int) -> None:
    """Lift the soft file limit to `hard`, or where that is unbounded to the largest the kernel
    grants, keeping the current one when it grants none of them."""
    asked = (hard,) if hard != resource.RLIM_INFINITY else _UNBOUNDED
    for soft in asked:
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (soft, hard))
        except ValueError, OSError:
            continue
        return


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
                _raise(cls._original[1])
            cls._holders += 1
        try:
            yield
        finally:
            with cls._lock:
                cls._holders -= 1
                if cls._holders == 0:
                    resource.setrlimit(resource.RLIMIT_NOFILE, cls._original)


class ProcessTree:
    """One started command and its descendants, stopped as a unit.

    grace: seconds a terminated tree is given before it is killed outright.
    """

    def __init__(self, process: subprocess.Popen[bytes], *, grace: float = 30.0) -> None:
        self.process = process
        self.grace = grace

    def wait(self, timeout: float | None = None) -> int | None:
        """The command's exit status, None when `timeout` seconds (None for no limit) pass first.

        A signal N reports `128 + N`, as a shell would, so an exit code reads the same whether a
        shell or this runner started the command.
        """
        try:
            returned = self.process.wait(timeout)
        except subprocess.TimeoutExpired:
            return None
        return 128 - returned if returned < 0 else returned

    def terminate(self) -> None:
        """Ask the whole tree to stop with SIGTERM."""
        for member in self._members():
            with suppress(psutil.NoSuchProcess):
                member.terminate()

    def kill(self) -> None:
        for member in self._members():
            with suppress(psutil.NoSuchProcess):
                member.kill()

    def stop(self) -> bool:
        """Terminate the tree, kill it when it outlives the grace, and say whether it had to be.

        The two-step end `timeout --kill-after` gave a job: SIGTERM first, a chance to flush.
        """
        self.terminate()
        if self.wait(self.grace) is not None:
            return False
        self.kill()
        self.wait()
        return True

    def _members(self) -> list[psutil.Process]:
        """The command and every living descendant, read before any is signalled, since ending a
        parent reparents its children out of reach."""
        try:
            root = psutil.Process(self.process.pid)
            return [*root.children(recursive=True), root]
        except psutil.NoSuchProcess:
            return []
