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
def test_the_scheduler_is_the_first_launcher_on_path_with_clusters_outranking_pueue(
    present: Sequence[str], expected: Scheduler, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A login node often carries pueue alongside the real scheduler, and a job belongs to the
    cluster there, so `sbatch` then `qsub` are looked for first and a bare host reports none."""
    monkeypatch.setattr(env_mod.shutil, "which", lambda name: name if name in present else None)
    assert Environment.probe().scheduler is expected
