import pytest

from mainboard.dispatch.schedulers import SCHEDULERS, Local, Pbs, Pueue, Scheduler, Slurm, pick
from mainboard.manifest import HostProfile


@pytest.mark.parametrize(
    ("kind", "expected"), [("pbs", Pbs), ("slurm", Slurm), ("ssh", Pueue), ("local", Local)]
)
def test_pick_maps_kind_to_scheduler(kind: str, expected: type) -> None:
    scheduler = pick(HostProfile(kind=kind))
    assert isinstance(scheduler, expected)
    assert isinstance(scheduler, Scheduler)


def test_pick_falls_back_to_ssh_for_an_unregistered_kind() -> None:
    assert isinstance(pick(HostProfile(kind="auto")), Pueue)
    assert isinstance(pick(HostProfile(kind="whatever")), Pueue)


def test_registry_names_are_stable() -> None:
    assert SCHEDULERS.names == ["pbs", "slurm", "ssh", "local"]
