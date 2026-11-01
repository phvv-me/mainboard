import ast
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

from mainboard.analysis.edits import Create, Edit, Remove
from mainboard.analysis.engine.apply import Plan
from mainboard.analysis.operations.python.imports import Place

from ...engine.builders import written
from .builders import module_of, path_of


def applied(sources: Mapping[str, str], edits: Sequence[Edit]) -> dict[str, str]:
    """Every file of `sources` after `edits` taken as one finding's, by the path it stands at then.

    A plan that defers the edits leaves every file as it was.
    """
    with tempfile.TemporaryDirectory() as folder:
        plan = Plan(written(Path(folder), sources))
        taken = plan.add(edits)
        edited = {path: data.decode() for path, data in plan.written().items()} if taken else {}
        return {
            (plan.moved(path) if taken else path): edited.get(path, text)
            for path, text in sources.items()
        }


def denoted_after(
    sources: Mapping[str, str], *, moves: Mapping[str, str], after: Mapping[str, str]
) -> dict[str, list[tuple[str, str]]]:
    """What the names each file of `sources` binds denote in `after`, by the file's old path.

    A module `moves` carries is read from the path its new name gives it.
    """
    found = {}
    for path in sources:
        module = module_of(path)
        landed = moves.get(module, module)
        where = path_of(landed) if module in moves else path
        found[path] = _denotations(where, module=landed, content=after[where])
    return found


def denoted_before(
    sources: Mapping[str, str], *, moves: Mapping[str, str]
) -> dict[str, list[tuple[str, str]]]:
    """What the names each file of `sources` binds denote once `moves` happened, by its path.

    A name denoting a moved module denotes its new name, and one denoting a name inside it, that
    name inside the new one.
    """
    found = {}
    for path, text in sources.items():
        denoting = _denotations(path, module=module_of(path), content=text)
        found[path] = sorted((name, _followed(moves, denoted)) for name, denoted in denoting)
    return found


def _denotations(path: str, *, module: str, content: str) -> list[tuple[str, str]]:
    """What the names a file's `from ... import` statements bind denote, as bound name and module.

    The file stands at `path` as the dotted `module`; a relative import is read from there.
    """
    here = Place(path, module)
    return sorted(
        (alias.asname or alias.name, f"{here.resolved(node)}.{alias.name}")
        for node in ast.walk(ast.parse(content))
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    )


def _followed(moves: Mapping[str, str], denoted: str) -> str:
    """What `denoted` stands for after `moves`: the new module, or a name inside the new one."""
    parent, _, leaf = denoted.rpartition(".")
    if denoted in moves:
        return moves[denoted]
    return f"{moves[parent]}.{leaf}" if parent in moves else denoted


def created(edits: Sequence[Edit]) -> list[str]:
    """The paths of the files `edits` create."""
    return [edit.path for edit in edits if isinstance(edit, Create)]


def removed(edits: Sequence[Edit]) -> list[str]:
    """The paths `edits` remove."""
    return [edit.path for edit in edits if isinstance(edit, Remove)]
