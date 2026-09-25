from pydantic import Field

from ...core.base import Declared


class QueuePolicy(Declared):
    """One scheduler queue's envelope, enforced at submit time.

    max_walltime: miyabi's `short-g` rejects exactly `08:00:00`, so its ceiling is `07:59:59`.
    mem_ceiling_gb: the cgroup memory ceiling actually accepted at submit.
    submittable: false for a router queue that is listed but not targetable.
    """

    max_walltime: str = ""
    mem_ceiling_gb: int = 0
    gpus_per_node: int = 0
    max_jobs: int = 0
    submittable: bool = True
    notes: str = ""

    def admits_walltime(self, walltime: str) -> bool:
        """Whether `walltime` (HH:MM:SS) fits under this queue's ceiling."""
        if not self.max_walltime:
            return True
        return _seconds(walltime) <= _seconds(self.max_walltime)


class Defaults(Declared):
    """A host's submit-time defaults, any of which a CLI flag overrides.

    mem_gb: an expression over `attempt` (1-based retry), so a retry escalates; so is `walltime`.
    gpu_name: the GPU type a metered provider rents; owned hosts ignore it.
    max_usd: the spend cap every provider backend refuses to submit without.
    vram_gb: card memory a job needs, flagged by `facts`, `compute`, `setup` and `center verify`
        on a host whose largest card holds less; zero declares no need.
    interact_queue: where `interact` goes instead of `queue`, for a site routing interactive
        allocations elsewhere (Miyabi's `interact-g` router).
    """

    queue: str = ""
    interact_queue: str = ""
    walltime: str = "00:30:00"
    mem_gb: str = ""
    gpus: int = Field(default=0, ge=0)
    vram_gb: int = Field(default=0, ge=0)
    gpu_name: str = ""
    max_usd: float = Field(default=0.0, ge=0.0)


def _seconds(walltime: str) -> int:
    hours, minutes, seconds = (int(part) for part in walltime.split(":"))
    return hours * 3600 + minutes * 60 + seconds
