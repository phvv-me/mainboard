from pathlib import Path

import pytest

from mainboard.state import Lake


@pytest.fixture
def lake(workspace: Path) -> Lake:
    """The workspace's lake, created and at this release's schema."""
    return Lake.at(workspace).ready()
