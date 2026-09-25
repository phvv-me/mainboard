import shutil

from patos import FrozenModel

from ..enums import Scheduler

# In priority order: a login node often carries pueue beside the cluster's own scheduler, and a
# job there belongs to the cluster.
_LAUNCHERS = (("sbatch", Scheduler.SLURM), ("qsub", Scheduler.PBS), ("pueue", Scheduler.PUEUE))


class Environment(FrozenModel):
    """The host's execution environment, the job scheduler available on PATH."""

    scheduler: Scheduler = Scheduler.NONE

    @classmethod
    def probe(cls) -> Environment:
        """Detect the job scheduler on PATH."""
        found = (scheduler for launcher, scheduler in _LAUNCHERS if shutil.which(launcher))
        return cls(scheduler=next(found, Scheduler.NONE))
