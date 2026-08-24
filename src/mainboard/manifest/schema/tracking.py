from enum import StrEnum, auto

from ...core.base import Declared


class TrackingMode(StrEnum):
    """Whether a tracked run ships live, queues on disk for a later drain, or is not tracked.

    `off` is the one setting that turns the lane off, so a workspace that says nothing at all
    still gets tracked. That default is deliberate: a run nobody recorded is a run nobody can
    compare later, and the cost of recording one is a file beside the receipts.
    """

    ONLINE = auto()
    OFFLINE = auto()
    OFF = auto()


class Tracking(Declared):
    """Where this workspace mirrors its receipts, beyond the files that already hold them.

    The receipts a job writes are the record and this table only says who else gets a copy,
    which is why nothing here can fail a job. It is read the way `[gates]` and `[templates]` are,
    as a decision the workspace makes rather than one the tool guesses, and it reaches no
    generated file, so declaring a project never stales an environment.

    Every field has a working default, so the table exists to tune the lane or to turn it off
    rather than to switch it on.

    provider: the registered sink the receipts are mirrored into.
    entity: the account or team the runs land under, the provider's own default when empty.
    project: the project the runs are grouped under there, the workspace's own name when empty.
    mode: `online` ships while the job runs, `offline` queues beside the receipts for a later
        drain (what a compute node with no egress needs), `off` mirrors nothing at all.
    interval: seconds between live machine samples, 0 for a job that samples nothing.
    """

    provider: str = "wandb"
    entity: str = ""
    project: str = ""
    mode: TrackingMode = TrackingMode.ONLINE
    interval: float = 10.0

    @property
    def on(self) -> bool:
        """Whether anything is mirrored at all: a declared provider, and a mode past off."""
        return bool(self.provider) and self.mode is not TrackingMode.OFF
