"""The no-scheduler backend: run the job straight through `bash` on the host, no daemon involved.

There is no queue and no persistent handle, so `submit` blocks until the job finishes and
`state` can only report a vanished post-mortem. Use `Pueue` instead whenever a daemon is
available, this is the bare fallback.
"""

from typing import TYPE_CHECKING

from ..shared import logger
from .base import JobState, Resources, read_log

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..transport import Machine


class Local:
    """Run jobs directly through `bash` on the host (no scheduler, no queue)."""

    name = "local"

    def cancel(self, remote: Machine, root: str, *, handle: str) -> None:
        del remote, root
        logger.info("local backend has no queue; cannot cancel %s", handle)

    def jobs(self, remote: Machine, root: str) -> list[JobState]:
        del remote, root
        return []

    def logs(self, remote: Machine, root: str, *, handle: str) -> str:

        return read_log(remote, root, handle=handle)

    def queues(self, remote: Machine, root: str) -> list[str]:
        del remote, root
        return []

    def revive(self, remote: Machine, root: str) -> list[str]:
        del remote, root
        raise SystemExit("the local backend runs bare bash; there is no daemon to revive")

    def state(self, remote: Machine, root: str, *, handle: str) -> JobState:
        del remote, root
        return JobState(handle=handle, state=None, exit_code=None, verdict="vanished")

    def states(self, remote: Machine, root: str, handles: Sequence[str]) -> dict[str, JobState]:
        del remote, root, handles
        return {}

    def stream(self, remote: Machine, root: str, *, handle: str) -> JobState:
        # `submit` already relayed the job's output in the foreground; nothing to follow.
        return self.wait(remote, root, handle=handle)

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

    def wait(self, remote: Machine, root: str, *, handle: str) -> JobState:
        del remote, root
        # `submit` ran the job to completion in the foreground and raised on a non-zero exit, so
        # a handle that reaches here finished fine; there is nothing to poll.
        return JobState(handle=handle, state="done", exit_code=0, verdict="ok")
