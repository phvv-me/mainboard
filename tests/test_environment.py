from __future__ import annotations

import pytest

from maquina import Environment, Machine, Scheduler
from maquina.models import MachineSnapshot
from maquina.models import environment as env_mod


@pytest.mark.parametrize(
    ("present", "expected"),
    [
        (("sbatch",), Scheduler.SLURM),
        (("qsub",), Scheduler.PBS),
        (("pueue",), Scheduler.PUEUE),
        (("sbatch", "pueue"), Scheduler.SLURM),  # cluster schedulers win over pueue
        (("qsub", "pueue"), Scheduler.PBS),
        ((), Scheduler.NONE),
    ],
)
def test_scheduler_detection(
    present: tuple[str, ...], expected: Scheduler, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The scheduler is read from PATH, with cluster schedulers taking priority over pueue."""
    monkeypatch.setattr(env_mod.shutil, "which", lambda name: name if name in present else None)
    assert env_mod._scheduler() == expected


def test_probe_fills_every_field(monkeypatch: pytest.MonkeyPatch) -> None:
    """probe() gathers user, primary group, all groups, and the scheduler."""
    monkeypatch.setattr(env_mod, "_user", lambda: "alice")
    monkeypatch.setattr(env_mod, "_primary_group", lambda: "research")
    monkeypatch.setattr(env_mod, "_all_groups", lambda: ("research", "gpu"))
    monkeypatch.setattr(env_mod, "_scheduler", lambda: Scheduler.SLURM)

    env = Environment.probe()

    assert env.user == "alice"
    assert env.group == "research"
    assert env.groups == ("research", "gpu")
    assert env.scheduler is Scheduler.SLURM


def test_group_helpers_tolerate_lookup_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed group lookup yields empty defaults instead of raising."""

    def boom(_gid: int) -> object:
        raise KeyError("no such gid")

    monkeypatch.setattr(env_mod.grp, "getgrgid", boom)
    assert env_mod._primary_group() == ""
    assert env_mod._all_groups() == ()


def test_snapshot_includes_environment_and_round_trips() -> None:
    """The machine snapshot carries the environment block and survives a JSON round-trip."""
    snap = Machine().snapshot()

    assert isinstance(snap.environment, Environment)
    restored = MachineSnapshot.model_validate_json(snap.model_dump_json())
    assert restored.environment == snap.environment
