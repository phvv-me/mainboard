from typing import TYPE_CHECKING

import pytest

from mainboard.lab import Run
from mainboard.lab.experiment import DeclaredExperiment

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def context(tmp_path: Path) -> Run:
    """A trial context whose `artifact_dir` is the test's tmp path, never the project cache."""
    return Run(model_id="gpt2", config=DeclaredExperiment(), artifact_dir=tmp_path)
