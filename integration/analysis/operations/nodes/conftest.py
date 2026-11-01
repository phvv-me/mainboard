from collections.abc import Callable, Iterator, Mapping
from functools import partial
from pathlib import Path

import pytest
from duckdb import DuckDBPyConnection

from mainboard.analysis.engine.facts import Budget
from mainboard.analysis.engine.lint import Lint, load_catalog, selected
from mainboard.analysis.operations.nodes import Dialect, Node, Site

from ...engine.builders import written
from .builders import grammar_nodes, sources


@pytest.fixture(scope="module")
def project(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A project holding the corpus the node tests parse."""
    return written(tmp_path_factory.mktemp("corpus"), sources())


@pytest.fixture(scope="module")
def cursor(project: Path) -> Iterator[DuckDBPyConnection]:
    """A cursor over the facts of the corpus, read by a Lint that fixes."""
    load_catalog()
    with Lint(project, selected(("ALL-CONT0002",))) as lint:
        yield lint.facts(project).cursor(None)


@pytest.fixture(scope="module")
def sites(cursor: DuckDBPyConnection) -> Mapping[str, Mapping[int, Site]]:
    """The site of every node of every corpus file, by path and node id, from one query."""
    rows = cursor.execute("SELECT path, node_id FROM ast_nodes ORDER BY path, node_id").fetchall()
    gathered = Site.gathered(cursor, rows)
    return {
        path: {node: site for (at, node), site in gathered.items() if at == path}
        for path in sources()
    }


@pytest.fixture(scope="module")
def grammar() -> Iterator[Callable[[type[Dialect], str], list[Node]]]:
    """Parse a text with a dialect's grammar, pruning nothing."""
    budget = Budget()
    with budget.connect() as db:
        yield partial(grammar_nodes, db)
    budget.close()
