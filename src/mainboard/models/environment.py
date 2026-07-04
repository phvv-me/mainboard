import shutil

from ..enums import Scheduler
from .base import FrozenModel
from .identity import Identity


def _scheduler() -> Scheduler:
    """Job scheduler on PATH; cluster schedulers take priority over pueue."""
    if shutil.which("sbatch"):
        return Scheduler.SLURM
    if shutil.which("qsub"):
        return Scheduler.PBS
    if shutil.which("pueue"):
        return Scheduler.PUEUE
    return Scheduler.NONE


class Environment(FrozenModel):
    """The host's execution environment: who is running and what scheduler is available.

    Probed from the OS and PATH so a tool can route work without re-detecting the
    user, group, or job scheduler on its own.

    user: login name of the current user.
    group: primary group name of the current user.
    groups: every group the current user belongs to.
    scheduler: the job scheduler found on PATH.
    """

    user: str = ""
    group: str = ""
    groups: tuple[str, ...] = ()
    scheduler: Scheduler = Scheduler.NONE

    @classmethod
    def probe(cls) -> Environment:
        """Detect the current user, group(s), and job scheduler."""
        identity = Identity.current()
        return cls(
            user=identity.user(),
            group=identity.primary_group(),
            groups=identity.all_groups(),
            scheduler=_scheduler(),
        )
