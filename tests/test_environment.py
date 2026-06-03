from __future__ import annotations

import pytest

from mainboard import Environment, Scheduler
from mainboard.models import environment as env_mod


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
    """`probe` gathers user, primary group, and all groups from the OS identity."""

    class FakeIdentity:
        def user(self) -> str:
            return "alice"

        def primary_group(self) -> str:
            return "research"

        def all_groups(self) -> tuple[str, ...]:
            return ("research", "gpu")

    monkeypatch.setattr(env_mod.Identity, "current", staticmethod(lambda: FakeIdentity()))
    monkeypatch.setattr(env_mod, "_scheduler", lambda: Scheduler.SLURM)

    env = Environment.probe()

    assert env.user == "alice"
    assert env.group == "research"
    assert env.groups == ("research", "gpu")
    assert env.scheduler is Scheduler.SLURM
