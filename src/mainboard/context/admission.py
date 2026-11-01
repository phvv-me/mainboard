from ..core.constants.units import BYTES_PER_GB, JOB_HOST_DISK
from ..core.errors import MissionError
from ..core.project import Project
from ..manifest.schema.compute.host import HostProfile


def admit_disk(job: str, *, host: str, root: str, free: int) -> None:
    """Refuse `job` when `host` has less free disk at its workspace than a job host keeps.

    A host filled to 99% stopped answering ssh at all (gold, 2026-10-31).
    free: bytes free at `root` now.
    """
    if free < JOB_HOST_DISK:
        raise MissionError(
            f"{job} needs {JOB_HOST_DISK / BYTES_PER_GB:.0f} GB free on {host} at {root}, which "
            f"has {free / BYTES_PER_GB:.1f} GB: `{Project().name} host evict {host} <path>` "
            "gives back what the lake already holds"
        )


def admit(profile: HostProfile, *, queue: str, walltime: str, mem_gb: int) -> None:
    """Refuse a submission the queue's declared policy would reject.

    The scheduler's own rejection arrives minutes later and cryptic; this one arrives before the
    ssh round-trip, naming the ceiling. An undeclared queue admits everything.

    walltime: the requested HH:MM:SS wall-clock limit.
    """
    policy = profile.policy(queue)
    if not policy.submittable:
        raise MissionError(f"queue {queue!r} is not submittable on this host: {policy.notes}")
    if not policy.admits_walltime(walltime):
        raise MissionError(
            f"walltime {walltime} exceeds the {queue!r} ceiling {policy.max_walltime}"
            + (f" ({policy.notes})" if policy.notes else "")
        )
    if policy.mem_ceiling_gb and mem_gb > policy.mem_ceiling_gb:
        raise MissionError(
            f"mem {mem_gb}GB exceeds the {queue!r} ceiling {policy.mem_ceiling_gb}GB"
        )
