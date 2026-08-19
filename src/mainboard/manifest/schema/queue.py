from pydantic import Field

from ...core.base import Declared


class QueuePolicy(Declared):
    """One scheduler queue's operational envelope, enforced at submit time.

    The typed home for what previously lived as prose: walltime ceilings
    (miyabi's `short-g` rejects exactly `08:00:00`, so its ceiling is
    `07:59:59`), the cgroup memory ceiling actually accepted at submit, and
    whether jobs may target the queue at all (router queues are listed but not
    submittable).
    """

    max_walltime: str = ""
    mem_ceiling_gb: int = 0
    gpus_per_node: int = 0
    max_jobs: int = 0
    submittable: bool = True
    notes: str = ""

    def admits_walltime(self, walltime: str) -> bool:
        """Whether `walltime` (HH:MM:SS) fits under this queue's ceiling.

        walltime: the requested wall-clock limit.
        """
        if not self.max_walltime:
            return True
        return _seconds(walltime) <= _seconds(self.max_walltime)


class Defaults(Declared):
    """A host's submit-time defaults, any of which a CLI flag overrides.

    `mem_gb` and `walltime` accept expressions over `attempt` (the 1-based
    retry number), evaluated at submit time, so a retried job escalates its
    request instead of dying to the same ceiling twice.

    `gpu_name` and `max_usd` are what a metered provider host needs and an
    owned one ignores: the GPU type to rent, and the spend cap every provider
    backend refuses to submit without, declared once per host rather than
    retyped on every submit.
    """

    queue: str = ""
    walltime: str = "00:30:00"
    mem_gb: str = ""
    gpus: int = Field(default=0, ge=0)
    gpu_name: str = ""
    max_usd: float = Field(default=0.0, ge=0.0)


def _seconds(walltime: str) -> int:
    hours, minutes, seconds = (int(part) for part in walltime.split(":"))
    return hours * 3600 + minutes * 60 + seconds
