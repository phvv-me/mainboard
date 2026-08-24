from collections.abc import Iterator

import pytest

from mainboard.observe import Store


@pytest.fixture(scope="module")
def store(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Store]:
    """One history database for the whole module, with every test keying its rows by job name.

    Opening a WAL SQLite file is the expensive half of a store test, so it happens once, and
    this block is also what exercises the context manager releasing the connection at the end.
    """
    with Store(tmp_path_factory.mktemp("history") / "history.sqlite") as opened:
        yield opened
