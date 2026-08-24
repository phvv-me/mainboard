import pytest

from mainboard.dispatch.schedulers import Local, Pbs, Pueue, Scheduler, Slurm, pick
from mainboard.dispatch.schedulers.registry import SCHEDULERS
from mainboard.manifest import HostProfile


@pytest.mark.parametrize(
    ("kind", "expected"),
    [
        ("pbs", Pbs),
        ("slurm", Slurm),
        ("ssh", Pueue),
        ("local", Local),
        ("auto", Pueue),
        ("whatever", Pueue),
    ],
)
def test_pick_maps_a_declared_kind_to_its_backend_and_falls_back_to_ssh(
    kind: str, expected: type
) -> None:
    """A host the manifest never pinned a scheduler for is a plain ssh box behind pueue."""
    scheduler = pick(HostProfile(kind=kind))
    assert isinstance(scheduler, expected)
    assert isinstance(scheduler, Scheduler)
    assert SCHEDULERS.names == ["pbs", "slurm", "ssh", "local"]
