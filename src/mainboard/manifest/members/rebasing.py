# A member's names and paths as the workspace spells them: each local path rebased onto the
# workspace, each task and paper namespaced by its member.

import posixpath
from collections import Counter
from pathlib import PurePosixPath
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from ..render.interpolate import Json
    from ..schema.environment import Task
    from .models import Member

# The tables whose entries are dependency specs, where a local `path` source can appear.
_SPEC_TABLES = frozenset({"deps", "dev", "dependency-overrides"})


def namespaced[T](
    declared: Mapping[str, Mapping[str, T]], *, root: Mapping[str, T]
) -> dict[str, T]:
    """Every member entry as `<member>:<name>`, and bare where nobody else takes the name.

    declared: each member's entries by member name.
    root: the root's own entries, which keep their names and win over any bare member one.
    """
    counts = Counter(name for entries in declared.values() for name in entries)
    named: dict[str, T] = {}
    for member, entries in declared.items():
        for name, entry in entries.items():
            named[f"{member}:{name}"] = entry
            if counts[name] == 1:
                named[name] = entry
    return named | dict(root)


def moved(
    member: Member, tasks: Mapping[str, Task], *, siblings: Mapping[str, Task]
) -> dict[str, Task]:
    """`member`'s tasks running from its directory, each dependency on a sibling renamed.

    siblings: every task of the member a dependency can name; any other stays the root's.
    """
    relocated: dict[str, Task] = {}
    for name, spec in tasks.items():
        task: dict[str, str | list[str] | dict[str, str]] = (
            {"run": spec} if isinstance(spec, str) else dict(spec)
        )
        if "run" in task:
            task["dir"] = rebased(member.path, path=str(task.get("dir", "")))
        if "depends" in task:
            task["depends"] = [
                f"{member.name}:{need}" if need in siblings else need for need in task["depends"]
            ]
        relocated[name] = task
    return relocated


def rebased(base: str, *, path: str) -> str:
    """`path`, relative to the member at `base`, as the workspace spells it; absolute ones stay."""
    if PurePosixPath(path).is_absolute():
        return path
    return posixpath.normpath(posixpath.join(base, path))


def sources_rebased(tree: Mapping[str, Json], base: str) -> dict[str, Json]:
    """`tree`, a member's manifest or a table in it, with every local dependency source rebased."""
    return {
        key: _specs_rebased(item, base)
        if key in _SPEC_TABLES
        else sources_rebased(item, base)
        if isinstance(item, dict)
        else item
        for key, item in tree.items()
    }


def _specs_rebased(table: Json, base: str) -> Json:
    """A dependency table, or a scope nested in one (`[dev]`), with its `path` sources rebased."""
    if not isinstance(table, dict):
        return table
    return {
        name: {**spec, "path": rebased(base, path=path)}
        if isinstance(spec, dict) and isinstance(path := spec.get("path"), str)
        else _specs_rebased(spec, base)
        for name, spec in table.items()
    }


def anchored(patterns: Iterable[str], base: str) -> tuple[str, ...]:
    """Gitignore patterns written inside the member at `base`, as the workspace root reads them.

    A slash before the end already anchors a pattern to its own directory; one without matches
    at any depth, which beneath the member is `**/`.
    """
    found: list[str] = []
    for pattern in patterns:
        negated, bare = ("!", pattern[1:]) if pattern.startswith("!") else ("", pattern)
        rooted = "/" in bare.rstrip("/")
        found.append(f"{negated}/{base}/{bare.lstrip('/') if rooted else f'**/{bare}'}")
    return tuple(found)
