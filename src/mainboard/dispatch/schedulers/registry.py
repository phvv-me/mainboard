from typing import TYPE_CHECKING

from patos import Strategy

from .local import Local
from .pbs import Pbs
from .pueue import Pueue
from .slurm import Slurm

if TYPE_CHECKING:
    from ...manifest.schema.host import HostProfile
    from .base import Scheduler

# The backend registry, built once at import: a resolved profile `kind` selects its scheduler.
# Adding a backend is one `register` line here.
SCHEDULERS: Strategy[Scheduler] = Strategy("scheduler")
SCHEDULERS.register("pbs", Pbs())
SCHEDULERS.register("slurm", Slurm())
SCHEDULERS.register("ssh", Pueue())
SCHEDULERS.register("local", Local())


def pick(profile: HostProfile) -> Scheduler:
    """The `Scheduler` for `profile` from its declared `kind`.

    `pbs` -> `Pbs`, `slurm` -> `Slurm`, `ssh` -> `Pueue` (the default ssh queue), `local` ->
    `Local` (bare bash, no daemon). An unknown or `auto` kind falls back to `ssh`, since a host
    the manifest never pinned a scheduler for is assumed to be a plain ssh box behind pueue.
    """
    return SCHEDULERS.select(profile.kind, default="ssh")
