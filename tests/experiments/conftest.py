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
    """A board rooted at the test's own tmp path, its dispatch registry held in memory."""
    return FakeBoard(tmp_path)


@pytest.fixture(scope="session")
def study() -> Study:
    """One study identity every fleet and report test dispatches its trials under.

    Session-scoped because a `Study` is frozen and every test that uses it writes its ledger
    under its own tmp path, so nothing is shared but the identity itself.
    """
    return Study.create("joint-search", config_space={"bits": [1, 2]}, git_sha="abc123")
