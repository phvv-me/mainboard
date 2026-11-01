from collections.abc import Generator
from pathlib import Path

import pytest

from mainboard import Board


@pytest.fixture
def board(workspace: Path) -> Generator[Board]:
    """The workspace's board, which owes no host anything."""
    found = Board(workspace)
    yield found
    found.dispatcher.cache.close()
