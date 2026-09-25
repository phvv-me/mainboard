# A command and everything it started, ended together.
#
# Signalling only the process a job spawned leaves its workers running: a dataloader, a compile
# worker pool or a serving engine outlives the `bash -c` that started it and keeps the card. The
# tree is read from the process table rather than from a process group, which is what makes the
# same code end a job on Linux, macOS and Windows, and the command stays in the runner's own
# group so a scheduler that signals the whole group (PBS, pueue) still reaches all of it directly.

import subprocess
from contextlib import suppress

import psutil


class ProcessTree:
    """One started command and its descendants, stopped as a unit.

    process: the command this tree is rooted at.
    grace: seconds a terminated tree is given before it is killed outright.
    """

    def __init__(self, process: subprocess.Popen[bytes], *, grace: float = 30.0) -> None:
        self.process = process
        self.grace = grace

    def wait(self, timeout: float | None = None) -> int | None:
        """The command's exit status once it ends, None when `timeout` seconds pass first.

        A command ended by a signal reports `128 + N`, the status a shell would have reported for
        it, so a job's exit code reads the same whether a shell or this runner started it.

        timeout: seconds to wait, None to wait for as long as it runs.
        """
        try:
            returned = self.process.wait(timeout)
        except subprocess.TimeoutExpired:
            return None
        return 128 - returned if returned < 0 else returned

    def terminate(self) -> None:
        """Ask the whole tree to stop, SIGTERM on POSIX and an immediate end on Windows."""
        for member in self._members():
            with suppress(psutil.NoSuchProcess):
                member.terminate()

    def kill(self) -> None:
        """End the whole tree outright."""
        for member in self._members():
            with suppress(psutil.NoSuchProcess):
                member.kill()

    def stop(self) -> bool:
        """Terminate the tree, kill it when it outlives the grace, and say whether it had to be.

        The two-step end `timeout --kill-after` gave a job: a command that honours SIGTERM gets
        the chance to flush and exit, and one that does not is killed once the grace runs out.
        """
        self.terminate()
        if self.wait(self.grace) is not None:
            return False
        self.kill()
        self.wait()
        return True

    def _members(self) -> list[psutil.Process]:
        """The command and every living descendant, read before any of them is signalled.

        Read first because ending a parent reparents its children, after which they can no
        longer be found under it. A member that ends while the list is being signalled is skipped
        by the caller, and a command already gone leaves nothing to signal.
        """
        try:
            root = psutil.Process(self.process.pid)
            return [*root.children(recursive=True), root]
        except psutil.NoSuchProcess:
            return []
