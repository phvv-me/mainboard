from datetime import UTC, datetime, timedelta

from mainboard.dispatch.backends import Rented
from mainboard.dispatch.state import RunRecord
from mainboard.settling.reaping.reaper import Kept, Stage

from .constants import NOW, PROVIDER


def rented(handle: str = "m-1", *, label: str = "mainboard-a", status: str = "running") -> Rented:
    """A machine the provider lists, created under `label`."""
    return Rented(handle=handle, label=label, status=status)


def kept(
    label: str = "mainboard-a",
    *,
    handle: str = "m-1",
    stage: Stage = Stage.DONE,
    age: timedelta = timedelta(hours=3),
    release_by: datetime | None = None,
) -> Kept:
    """A creation this center minted through the provider `age` before `NOW`, at `stage`."""
    return Kept(
        label=label,
        target=PROVIDER,
        provider=PROVIDER,
        handle=handle,
        stage=stage,
        submitted=NOW - age,
        release_by=release_by,
    )


def machine(handle: str, *, label: str) -> dict:
    """One instance in Shadeform's listing."""
    return {"id": handle, "name": label, "status": "active", "ip": "203.0.113.7"}


def record(label: str, *, handle: str, verdict: str, reported: str | None = None) -> RunRecord:
    """A run the registry holds for the creation `label`, dispatched three hours ago by the clock.

    The pass rules at the real instant, so this age is real too and `NOW` plays no part.
    """
    return RunRecord(
        handle=handle,
        target=PROVIDER,
        kind=PROVIDER,
        script="train.py",
        args="",
        submitted_at=(datetime.now(UTC) - timedelta(hours=3)).isoformat(),
        creation=label,
        verdict=verdict,
        reported=reported,
    )
