from enum import StrEnum, auto

from ...core.base import Declared


class TrackingMode(StrEnum):
    """Whether a tracked run ships live, queues on disk for a later drain, or is not tracked.

    Only `off` turns tracking off: an unrecorded run cannot be compared later, and recording
    costs a file beside the receipts.
    """

    ONLINE = auto()
    OFFLINE = auto()
    OFF = auto()


class Tracking(Declared):
    """Where this workspace mirrors its receipts, which stay the record, so it never fails a job.

    entity: the account or team, the provider's default when empty.
    project: the workspace's own name when empty.
    mode: `offline` queues beside the receipts for a later drain, for nodes with no egress.
    interval: seconds between live machine samples, 0 for none.
    """

    provider: str = "wandb"
    entity: str = ""
    project: str = ""
    mode: TrackingMode = TrackingMode.ONLINE
    interval: float = 10.0

    @property
    def on(self) -> bool:
        """Whether anything is mirrored: a declared provider and a mode other than off."""
        return bool(self.provider) and self.mode is not TrackingMode.OFF
