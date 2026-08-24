"""The no-scheduler backend: run the job straight through `bash` on the host, no daemon involved.

There is no queue and no persistent handle, so `submit` blocks until the job finishes and
`state` can only report a vanished post-mortem. Use `Pueue` instead whenever a daemon is
available, this is the bare fallback.
"""

from typing import TYPE_CHECKING

from ..shared import logger
from ..vocabulary import JobState, Resources
from .base import read_log, workspace_session

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..transport import Machine


class Local:
    """Run jobs directly through `bash` on the host (no scheduler, no queue)."""

    name = "local"

    def cancel(self, remote: Machine, root: str, *, handle: str) -> None:
        del remote, root
        logger.info("local backend has no queue; cannot cancel %s", handle)

    def interactive(self, *, env: str, command: Sequence[str], resources: Resources) -> str:
        return workspace_session(env=env, command=command, resources=resources)

    def logs(self, remote: Machine, root: str, *, handle: str) -> str:
        return read_log(remote, root, handle=handle)

    def state(self, remote: Machine, root: str, *, handle: str) -> JobState:
        del remote, root
        return JobState(handle=handle, state=None, exit_code=None, verdict="vanished")

    def states(self, remote: Machine, root: str, handles: Sequence[str]) -> dict[str, JobState]:
        """Every requested handle, vanished, since a queue that keeps nothing remembers nothing.

        Answered here rather than left absent so a caller batching a whole host's handles never
        falls back to a per-handle probe that would reach the same conclusion one job at a time.
        """
        return {handle: self.state(remote, root, handle=handle) for handle in handles}

    def submit(
        self,
        remote: Machine,
        root: str,
        *,
        script: str,
        args: Sequence[str],
        resources: Resources,
    ) -> str:
        del root, resources
        remote["bash"][[script, *args]]()
        return script
