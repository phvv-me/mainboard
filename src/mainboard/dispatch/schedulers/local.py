"""The no-scheduler backend: run the job script straight through `sh` on the host, no daemon.

With no queue and no persistent handle, `submit` blocks until the job finishes and `state` can
only report a vanished post-mortem. The bare fallback: use `Pueue` wherever a daemon runs.
"""

import shlex
from typing import TYPE_CHECKING

from ..shared import logger
from ..vocabulary import JobState, Resources
from .base import read_log, within, workspace_session

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..transport import Machine


class Local:
    """Run job scripts directly through `sh` on the host (no scheduler, no queue)."""

    name = "local"

    def cancel(self, remote: Machine, root: str, *, handle: str) -> None:
        logger.info("local backend has no queue; cannot cancel %s", handle)

    def interactive(self, *, env: str, command: Sequence[str], resources: Resources) -> str:
        return workspace_session(env=env, command=command, resources=resources)

    def logs(self, remote: Machine, root: str, *, handle: str) -> str:
        return read_log(remote, root, handle=handle)

    def state(self, remote: Machine, root: str, *, handle: str) -> JobState:
        return JobState(handle=handle, verdict="vanished")

    def states(self, remote: Machine, root: str, handles: Sequence[str]) -> dict[str, JobState]:
        """Every handle, vanished: answered rather than left absent, so no per-handle re-probe."""
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
        remote["bash"][["-lc", within(root, shlex.join(["sh", script, *args]))]]()
        return script
