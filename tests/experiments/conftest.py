from pathlib import Path

import pytest

from mainboard.dispatch.state import Cache
from mainboard.experiments import Study

from .support import FakeBoard, dispatch_cache


@pytest.fixture
def cache() -> Cache:
    """A dispatch run registry a study's report is joined against."""
    return dispatch_cache()


@pytest.fixture
def board(tmp_path: Path) -> FakeBoard:
    """A board rooted at the test's tmp path, its dispatch registry in memory."""
    return FakeBoard(tmp_path)


@pytest.fixture(scope="session")
def study() -> Study:
    """One frozen study identity; each test writes its ledger under its own tmp path."""
    return Study.create("joint-search", config_space={"bits": [1, 2]}, source_digest="abc123")
