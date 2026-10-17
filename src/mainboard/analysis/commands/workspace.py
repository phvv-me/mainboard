"""A project's syntax-tree facts, opened once for a command's SQL."""

import atexit
from collections.abc import Mapping, Sequence
from hashlib import sha1
from pathlib import Path

from ...log import logger
from ..catalog import needed
from ..facts import Budget, Facts, Query, listed
from ..providers.ducks import Ducks

CACHES = Path.home() / ".cache/mainboard/analysis"

type Binding = str | int | Sequence[str]


class Workspace:
    """The `ast_files` and `ast_nodes` views of one project root, parsed by sitting_duck, and the
    stored relations commands install over them.

    The parsed trees stay in a DuckDB file per project (`Ducks`' `astcache` tables, keyed by file
    content), so a command after the first parses only the files that changed; trees of contents
    no longer in the tree are dropped on each open. A command running while another holds the
    file keeps its trees in its own fact base instead.

    Relations read Python as `python.ast_nodes` and `python.ast_files`; a project without Python
    gets empty ones rather than an error.
    """

    def __init__(self, root: Path, *, memory: str = "2GB") -> None:
        self.root = root.resolve()
        self.installed: set[str] = set()
        folder = CACHES / sha1(str(self.root).encode(), usedforsecurity=False).hexdigest()[:12]
        budget = Budget(memory=memory)
        self.facts = Facts(self.root, budget)
        atexit.register(budget.close)
        atexit.register(self.facts.close)
        self.db = self.facts.db.cursor()
        Ducks(cache=folder / "ast.duckdb", budget=budget, exclusive=True).register(
            self.facts, self.root
        )
        if "python" not in self.facts.languages:
            self.db.execute("CREATE SCHEMA IF NOT EXISTS python")
            for view in ("ast_files", "ast_nodes"):
                self.db.execute(
                    f"CREATE VIEW python.{view} AS SELECT * FROM main.{view} WHERE false"
                )

    def install(self, *sql: str) -> None:
        """Define the relations `sql` calls, each after those it calls, and compute the stored
        ones once, so that `FROM relation()` reads a table."""
        for relation in needed(Query(sql=text) for text in sql):
            if relation.name() in self.installed:
                continue
            if relation.stored:
                self.db.execute(
                    f"CREATE OR REPLACE TABLE main.{relation.name()}__rows AS {relation.query}"
                )
            self.db.execute(relation.definition())
            self.installed.add(relation.name())
            logger.debug("installed {}", relation.name())

    def rows(self, sql: str, parameters: Mapping[str, Binding] | None = None) -> list[dict]:
        """The rows of `sql` as dicts, after installing the relations it calls."""
        self.install(sql)
        cursor = self.db.execute(sql, parameters or {})
        names = [column for column, *_ in cursor.description]
        return [dict(zip(names, row, strict=True)) for row in cursor.fetchall()]

    def files(self) -> list[str]:
        """Every file of the project git does not ignore."""
        return listed(self.root)

    def read_texts(self, table: str, paths: list[str]) -> None:
        """Create `table(path, content)` holding the text of `paths`, project-relative."""
        self.db.execute(f"CREATE OR REPLACE TABLE {table} (path VARCHAR, content VARCHAR)")
        if paths:
            self.db.execute(
                f"INSERT INTO {table} SELECT path, content FROM "
                "(SELECT unnest($paths) AS path, unnest($files) AS file) "
                "JOIN read_text($files) r ON r.filename = file",
                {"paths": paths, "files": [str(self.root / path) for path in paths]},
            )
