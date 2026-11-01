# A command and everything it started, ended together.
#
# Signalling only the process a job spawned leaves its workers (a dataloader, a compile pool, a
# serving engine) running and holding the card. The tree is read from the process table rather
# than a process group, so the same code ends a job on Linux and macOS, and the command stays in
# the runner's group so a scheduler signalling the group (PBS, pueue) still reaches it.

import subprocess
from contextlib import suppress

import psutil


class ProcessTree:
    """One started command and its descendants, stopped as a unit.

    grace: seconds a terminated tree is given before it is killed outright.
    """

    @staticmethod
    def end(pid: int, *, force: bool = False) -> bool:
        """Terminate `pid` and everything it started, children first, or kill them with `force`.

        The tree is read before any member is signalled, since ending a parent reparents its
        children out of reach. Answers whether `pid` was there to end.
        """
        with suppress(psutil.NoSuchProcess):
            root = psutil.Process(pid)
            for member in [*reversed(root.children(recursive=True)), root]:
                with suppress(psutil.Error):
                    (member.kill if force else member.terminate)()
            return True
        return False

    def __init__(self, process: subprocess.Popen[bytes], *, grace: float = 30.0) -> None:
        self.process = process
        self.grace = grace

    def kill(self) -> None:
        self.end(self.process.pid, force=True)

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

    def terminate(self) -> None:
        """Ask the whole tree to stop with SIGTERM."""
        self.end(self.process.pid)

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
