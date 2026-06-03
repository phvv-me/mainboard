from __future__ import annotations

import pytest

from maquina import Environment, Scheduler
from maquina.models import environment as env_mod


@pytest.mark.parametrize(
    ("present", "expected"),
    [
        (("sbatch",), Scheduler.SLURM),
        (("qsub",), Scheduler.PBS),
        (("pueue",), Scheduler.PUEUE),
        (("sbatch", "pueue"), Scheduler.SLURM),
        (("qsub", "pueue"), Scheduler.PBS),
        ((), Scheduler.NONE),
    ],
)
def test_scheduler_priority(
    present: tuple[str, ...], expected: Scheduler, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The scheduler is read from PATH, cluster schedulers winning over pueue."""
    monkeypatch.setattr(env_mod.shutil, "which", lambda name: name if name in present else None)
    assert env_mod._scheduler() == expected


def test_probe_fills_every_field(monkeypatch: pytest.MonkeyPatch) -> None:
    """`probe` gathers user, primary group, all groups, and the scheduler."""
    monkeypatch.setattr(env_mod, "_user", lambda: "alice")
    monkeypatch.setattr(env_mod, "_primary_group", lambda: "research")
    monkeypatch.setattr(env_mod, "_all_groups", lambda: ("research", "gpu"))
    monkeypatch.setattr(env_mod, "_scheduler", lambda: Scheduler.SLURM)

    env = Environment.probe()

    assert env.user == "alice"
    assert env.group == "research"
    assert env.groups == ("research", "gpu")
    assert env.scheduler is Scheduler.SLURM


def test_user_tolerates_lookup_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed username lookup yields an empty string instead of raising."""

    def boom() -> str:
        raise OSError("no login name")

    monkeypatch.setattr(env_mod.getpass, "getuser", boom)
    assert env_mod._user() == ""


def test_group_helpers_tolerate_lookup_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed group lookup yields empty defaults instead of raising."""

    def boom(_gid: int) -> object:
        raise KeyError("no such gid")

    monkeypatch.setattr(env_mod.grp, "getgrgid", boom)
    assert env_mod._primary_group() == ""
    assert env_mod._all_groups() == ()


def test_group_helpers_resolve_names(monkeypatch: pytest.MonkeyPatch) -> None:
    """Successful gid lookups return the resolved group names."""
    names = {10: "research", 20: "gpu"}
    monkeypatch.setattr(env_mod.os, "getgid", lambda: 10)
    monkeypatch.setattr(env_mod.os, "getgroups", lambda: [10, 20])
    monkeypatch.setattr(
        env_mod.grp, "getgrgid", lambda gid: type("G", (), {"gr_name": names[gid]})()
    )
    assert env_mod._primary_group() == "research"
    assert env_mod._all_groups() == ("research", "gpu")
