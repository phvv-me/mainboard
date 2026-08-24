from typing import TYPE_CHECKING

import pytest

from mainboard import Project, load

if TYPE_CHECKING:
    from pathlib import Path

    from mainboard import Manifest


@pytest.fixture
def loaded(workspace: Path) -> Manifest:
    """The full-featured fixture manifest, rendered and validated straight off disk."""
    return load(workspace / Project().manifest)
