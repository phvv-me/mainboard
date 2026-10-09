from typing import TYPE_CHECKING

from patos import Strategy

from .held import Held
from .local import Local
from .pbs import Pbs
from .pueue import Pueue
from .slurm import Slurm

if TYPE_CHECKING:
    from ...manifest.schema.host import HostProfile
    from .base import Scheduler

# A resolved profile `kind` selects its scheduler; adding a backend is one `register` line.
SCHEDULERS: Strategy[Scheduler] = Strategy("scheduler")
SCHEDULERS.register("pbs", Pbs())
SCHEDULERS.register("held", Held())
SCHEDULERS.register("slurm", Slurm())
SCHEDULERS.register("ssh", Pueue())
SCHEDULERS.register("local", Local())


def kind_of(profile: HostProfile, queue: str = "") -> str:
    """The scheduler kind a submission to `queue` runs under: the queue's declared scheduler
    (`held`), else the host's kind. The one kind a dispatch records, so every later probe, log
    read and cancel reaches the scheduler that took the job."""
    return profile.policy(queue).scheduler or profile.kind


def pick(profile: HostProfile, queue: str = "") -> Scheduler:
    """The `Scheduler` for a submission to `queue`, falling back to `ssh` for an unknown or
    `auto` kind.

    A host the manifest never pinned a scheduler for is assumed to be a plain ssh box behind pueue.
    """
    return SCHEDULERS.select(kind_of(profile, queue), default="ssh")
