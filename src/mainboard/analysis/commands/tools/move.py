# Moving top-level definitions between a project's modules, every import of them following.

import difflib
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath

from ....log import logger
from ...engine.apply import Plan
from ...engine.facts import Query
from ...engine.lint import Lint, selected
from ...operations import Operation

_ROW = """
    WITH wanted AS (SELECT unnest($names) AS name)
    SELECT f.module, s.content,
           (SELECT list(x ORDER BY x.path, x.start_line, x.start_column) FROM imported_names() x
            WHERE x.module = f.module AND x.name IN (SELECT name FROM wanted)) AS importers,
           (SELECT list(x ORDER BY x.path, x.statement_start_line, x.start_line, x.start_column)
            FROM reached_names() x
            WHERE x.module = f.module AND x.name IN (SELECT name FROM wanted)) AS reached,
           (SELECT d.module FROM core_file d WHERE d.path = $destination) AS destination_module,
           (SELECT d.content FROM source_files d WHERE d.path = $destination)
             AS destination_content
    FROM core_file f JOIN source_files s USING (path)
    WHERE f.path = $path
"""


def move(names: tuple[str, ...], *, to: Path, root: Path = Path(), write: bool = False) -> int:
    """Move the top-level definitions NAMES (`path.py::Name`, one module) into the module TO.

    Each goes with its decorators and comments, with the imports it reads, and every import of it
    in the project is respelled; the old module imports back what it still reads. Without
    `--write` the change prints as a diff. The exit is 1 when the move is refused, with the
    reason: a definition reading a name of its module that stays, a cycle, a destination
    defining the name, names from more than one module.

    Args:
        names: the definitions, `src/pkg/module.py::Name`, all from one module.
        to: the module they move to, created when missing.
        root: the project holding both.
        write: write the change instead of printing it.
    """
    root = root.resolve()
    destination = str(PurePosixPath((root / to).resolve().relative_to(root)))
    row = _row(root, names, destination=destination)
    operation, plan = Operation.find("move-definitions")(), Plan(root)
    if row is None or (edits := operation.edits(row)) is None or not plan.add(edits):
        logger.result(
            f"refused: {operation.why or 'names of one module of the project, no file sealed'}"
        )
        return 1
    for name, data in sorted(plan.written().items()):
        _shown(root, name, data, before=plan.source(name).data, write=write)
    return 0


def _module(row: Mapping, destination: str) -> str:
    """The dotted name a new module at `destination` takes, read off its source's path and name."""
    parts = PurePosixPath(row["path"]).with_suffix("").parts
    depth = len(row["module"].split(".")) - (parts[-1] == "__init__")
    spelled = PurePosixPath(destination).with_suffix("").parts[len(parts) - depth :]
    return ".".join(part for part in spelled if part != "__init__")


def _row(root: Path, names: Sequence[str], *, destination: str) -> dict | None:
    """The `move-definitions` row for `names`, None unless they name one module of `root`."""
    paths = {name.partition("::")[0] for name in names}
    wanted = [name.partition("::")[2] for name in names]
    if len(paths) != 1:
        return None
    path = paths.pop()
    with Lint(root, selected(())) as lint:
        facts = lint.facts(root)
        facts.ensure(Query(sql=_ROW))
        result = facts.cursor(None).execute(
            _ROW, {"names": wanted, "path": path, "destination": destination}
        )
        columns, found = [column for column, *_ in result.description], result.fetchone()
    if found is None:
        return None
    row = dict(zip(columns, found, strict=True)) | {
        "path": path,
        "names": wanted,
        "destination": destination,
    }
    return row | {"destination_module": row["destination_module"] or _module(row, destination)}


def _shown(root: Path, name: str, data: bytes, *, before: bytes, write: bool) -> None:
    """Write the file `name` as `data`, or print how it would change from `before`."""
    if write:
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_bytes(data)
        logger.result(f"wrote {name}")
        return
    lines = before.decode().splitlines(keepends=True), data.decode().splitlines(keepends=True)
    logger.result("".join(difflib.unified_diff(*lines, f"a/{name}", f"b/{name}")))
