from ..core.errors import MissionError
from ..manifest.schema.host import HostProfile


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
