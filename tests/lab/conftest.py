from typing import TYPE_CHECKING

import pytest

from mainboard.lab import Run
from mainboard.lab.experiment import DeclaredExperiment

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def context(tmp_path: Path) -> Run:
    """The per-trial context a gate check, a measure call and an artifact write are handed.

    Carries a scratch `artifact_dir` under the test's own tmp path, so nothing here ever
    reaches the project cache a real trial would write into.
    """
    return Run(model_id="gpt2", config=DeclaredExperiment(), artifact_dir=tmp_path)
