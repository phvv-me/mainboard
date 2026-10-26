from typing import TYPE_CHECKING

from patos import Strategy

from .local import Local
from .pbs.core import Pbs
from .pbs.held import Held
from .pueue import Pueue
from .slurm import Slurm

if TYPE_CHECKING:
    from ...manifest.schema.compute.host import HostProfile
    from .base import Scheduler

# A resolved profile `kind` selects its scheduler; adding a backend is one `register` line.
schedulers: Strategy[Scheduler] = Strategy("scheduler")
schedulers.register("pbs", Pbs())
schedulers.register("held", Held())
schedulers.register("slurm", Slurm())
schedulers.register("ssh", Pueue())
schedulers.register("local", Local())


def kind_of(profile: HostProfile, queue: str = "") -> str:
    """The scheduler kind a submission to `queue` runs under.

    That is the queue's declared scheduler (`held`), else the host's kind. A dispatch records this
    one kind, so every later probe, log read and cancel reaches the scheduler that took the job.
    """
    return profile.policy(queue).scheduler or profile.kind


def pick(profile: HostProfile, queue: str = "") -> Scheduler:
    """The `Scheduler` for a submission to `queue`.

    An unknown or `auto` kind falls back to `ssh`.

    A host the manifest never pinned a scheduler for is assumed to be a plain ssh box behind pueue.
    """
    return schedulers.select(kind_of(profile, queue), default="ssh")
