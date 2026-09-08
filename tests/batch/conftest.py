from pathlib import Path

import pytest

from mainboard import Board
from mainboard.dispatch import Dispatcher, GitignoreFilter
from mainboard.dispatch.state import Cache

from .support import Recorder


@pytest.fixture
def lab(workspace: Path) -> Board:
    """A board whose registry and settlement lock stay inside a throwaway workspace.

    The fixture manifest's own hosts, so a batch resolves real profiles, real sync scopes and
    real queue policies, including the file-backed registry needed for durable settlement.
    """
    board = Board(workspace)
    board.shared["dispatcher"] = Dispatcher(
        cache=Cache(workspace / "dispatch.sqlite"), sync=GitignoreFilter(workspace)
    )
    return board


@pytest.fixture
def bus() -> Recorder:
    """The in-memory receipts a batch publishes to when a test wants to read them back."""
    return Recorder()
