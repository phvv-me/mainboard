# Moving top-level definitions between a project's modules, every import of them following.

import difflib
from collections.abc import Sequence
from pathlib import Path, PurePosixPath

from ....log import logger
from ...catalog import define, needed
from ...engine.apply import Plan
from ...engine.facts import Facts, Query
from ...engine.lint import Lint, selected
from ...operations import Operation

_ROW = "SELECT * FROM move_row($path, $names, $destination)"


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


def _installed(facts: Facts) -> None:
    """Install `_ROW`'s relations and functions, which the enabled rules may not call."""
    relations = needed([Query(sql=_ROW)])
    define(facts, [_ROW, *(relation.query for relation in relations)])
    for relation in relations:
        facts.install(relation, languages={None})


def _row(root: Path, names: Sequence[str], *, destination: str) -> dict | None:
    """The `move-definitions` row for `names`, None unless they name one module of `root`."""
    paths = {name.partition("::")[0] for name in names}
    if len(paths) != 1:
        return None
    arguments = {
        "path": paths.pop(),
        "names": [name.partition("::")[2] for name in names],
        "destination": destination,
    }
    with Lint(root, selected(())) as lint:
        facts = lint.facts(root)
        _installed(facts)
        result = facts.cursor(None).execute(_ROW, arguments)
        columns, found = [column for column, *_ in result.description], result.fetchone()
    return None if found is None else dict(zip(columns, found, strict=True))


def _shown(root: Path, name: str, data: bytes, *, before: bytes, write: bool) -> None:
    """Write the file `name` as `data`, or print how it would change from `before`."""
    if write:
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_bytes(data)
        logger.result(f"wrote {name}")
        return
    lines = before.decode().splitlines(keepends=True), data.decode().splitlines(keepends=True)
    logger.result("".join(difflib.unified_diff(*lines, f"a/{name}", f"b/{name}")))
