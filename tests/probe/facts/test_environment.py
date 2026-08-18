from typing import TYPE_CHECKING

import pytest

from mainboard.probe import Environment, Scheduler
from mainboard.probe.facts import environment as env_mod

if TYPE_CHECKING:
    from collections.abc import Sequence


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
    present: Sequence[str], expected: Scheduler, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The scheduler is read from PATH, cluster schedulers winning over pueue."""
    monkeypatch.setattr(env_mod.shutil, "which", lambda name: name if name in present else None)
    assert env_mod.detect_scheduler() == expected


def test_probe_reads_the_scheduler(monkeypatch: pytest.MonkeyPatch) -> None:
    """`probe` builds an `Environment` from the detected scheduler."""
    monkeypatch.setattr(env_mod, "detect_scheduler", lambda: Scheduler.SLURM)
    assert Environment.probe().scheduler is Scheduler.SLURM
