import ast
import tempfile
from collections.abc import Generator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import NamedTuple

from duckdb import DuckDBPyConnection

from mainboard.analysis.edits import Edit
from mainboard.analysis.engine.lint import Lint, load_catalog
from mainboard.analysis.operations.python.imports import Statement
from mainboard.analysis.rules.python.exceptions.core import SharedExceptionPlacement

from ...engine.builders import written
from ..modules.builders import module_of
from ..modules.readers import applied

type Row = dict[str, str | int | list[dict[str, str | int | list[str]]] | None]

_LOCAL = "src/pkg/m.py"
_PACKAGE = {
    "pyproject.toml": '[project]\nname = "pkg"\nversion = "0"\n',
    "src/pkg/__init__.py": "",
}
_READERS = """
    SELECT s.content, r.*
    FROM source_files s, name_readers($module, [$name]) r
    WHERE s.path = $path
"""


class Case(NamedTuple):
    """A module of a scratch package renamed, and what the rename makes of the package.

    files: the sources by file name; the defining module is `a.py`, and `CASE` is the name of the
        case's own package.
    after: the sources the rename changes, by file name; none for a refused rename.
    why: the part of the reason a refused rename is given.
    """

    files: dict[str, str]
    after: dict[str, str]
    why: str = ""
    name: str = "_hidden"
    to: str = "hidden"


def code(*lines: str) -> str:
    """The source of `lines`, each ending with a newline."""
    return "".join(f"{line}\n" for line in lines)


def local_row(source: str, *, name: str = "ab", to: str = "row_count") -> Row:
    """The `rewrite-references` row renaming local `name` of the first function of `source`."""
    opening = ("def ", "async def ")
    line = next(
        n for n, text in enumerate(source.splitlines(), 1) if text.lstrip().startswith(opening)
    )
    return {
        "operation": "rewrite-references",
        "scope": "function",
        "path": _LOCAL,
        "content": source,
        "line": line,
        "name": name,
        "to": to,
    }


def rewritten(source: str, edits: Sequence[Edit]) -> str:
    """`source`, the one file of a scratch project, once `edits` are applied to it."""
    return applied({_LOCAL: source}, edits)[_LOCAL]


def statement_at(source: str, line: int = 1) -> Statement:
    """The `from ... import` statement opening on `line` of `source`, quoted as the facts do."""
    node = next(
        each
        for each in ast.walk(ast.parse(source))
        if isinstance(each, ast.ImportFrom) and each.lineno == line
    )
    lines = source.split("\n")[node.lineno - 1 : node.end_lineno]
    record = {
        "path": _LOCAL,
        "importer": "pkg.m",
        "statement": "\n".join(lines),
        "start_line": node.lineno,
        "start_column": node.col_offset,
        "end_line": node.end_lineno,
        "end_column": node.end_col_offset,
    }
    return Statement(record, destination="pkg.a")


@contextmanager
def facts_cursor(sources: Mapping[str, str]) -> Generator[DuckDBPyConnection]:
    """A cursor over the facts of a scratch project holding the package and `sources`.

    The facts carry `name_readers`, which a rule whose fix reads `move_row` brings along.
    """
    load_catalog()
    with tempfile.TemporaryDirectory(prefix="mb-names-") as scratch:
        root = written(Path(scratch), {**_PACKAGE, **sources})
        with Lint(root, (SharedExceptionPlacement,)) as lint:
            yield lint.facts(root).cursor(None)


def module_row(
    cursor: DuckDBPyConnection,
    *,
    path: str,
    name: str,
    to: str,
    misstating: Mapping[str, str] | None = None,
) -> Row:
    """The `rewrite-references` row renaming the module-level `name` of the file at `path`.

    The file and its readers come from `cursor`, the facts of a scratch project; the columns of
    `misstating` replace those of every importer record, to state what the facts never would.
    """
    module = module_of(path)
    result = cursor.execute(_READERS, {"module": module, "name": name, "path": path})
    columns = [column for column, *_ in result.description]
    readers = dict(zip(columns, result.fetchone() or [], strict=True))
    readers["importers"] = [{**each, **(misstating or {})} for each in readers["importers"] or []]
    return {
        "operation": "rewrite-references",
        "scope": "module",
        "path": path,
        "module": module,
        "name": name,
        "to": to,
        **readers,
    }
