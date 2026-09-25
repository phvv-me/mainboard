from typing import TYPE_CHECKING

from patos import Strategy

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
SCHEDULERS.register("slurm", Slurm())
SCHEDULERS.register("ssh", Pueue())
SCHEDULERS.register("local", Local())


def pick(profile: HostProfile) -> Scheduler:
    """The `Scheduler` for `profile.kind`, falling back to `ssh` for an unknown or `auto` kind.

    A host the manifest never pinned a scheduler for is assumed to be a plain ssh box behind pueue.
    """
    return SCHEDULERS.select(profile.kind, default="ssh")
