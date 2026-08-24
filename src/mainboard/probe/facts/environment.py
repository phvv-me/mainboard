import shutil

from patos import FrozenModel

from ..enums import Scheduler


class Environment(FrozenModel):
    """The host's execution environment, the job scheduler available on PATH.

    scheduler: the job scheduler found on PATH.
    """

    scheduler: Scheduler = Scheduler.NONE

    @classmethod
    def probe(cls) -> Environment:
        """Detect the job scheduler on PATH."""
        return cls(scheduler=Environment._detect_scheduler())

    @staticmethod
    def _detect_scheduler() -> Scheduler:
        """Job scheduler on PATH, with cluster schedulers taking priority over pueue."""
        if shutil.which("sbatch"):
            return Scheduler.SLURM
        if shutil.which("qsub"):
            return Scheduler.PBS
        if shutil.which("pueue"):
            return Scheduler.PUEUE
        return Scheduler.NONE
