from ..core.errors import MissionError
from ..manifest.schema.host import HostProfile


def admit(profile: HostProfile, *, queue: str, walltime: str, mem_gb: int) -> None:
    """Refuse a submission the queue's declared policy would reject.

    The scheduler's own rejection arrives minutes later with a cryptic
    message; this one arrives before the ssh round-trip, naming the ceiling.
    A queue the profile does not declare admits everything.

    profile: the resolved host profile.
    queue: the queue being targeted.
    walltime: the requested HH:MM:SS wall-clock limit.
    mem_gb: the requested memory in gigabytes.
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
