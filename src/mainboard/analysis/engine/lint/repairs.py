"""A fix's edit rows as edits, grouped by the finding each one repairs and placed where it is."""

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING

from ...edits import Append, Create, Edit, Import, Remove, Rename, Replace
from ...findings import Hit, Span
from ...operations import Operation
from ..facts import snake

if TYPE_CHECKING:
    from duckdb import DuckDBPyConnection

    from ...catalog import Rule

# The operations a fix row states on whole files.
_FILED = frozenset({"append", "create", "remove", "rename"})

# Each kind of edit as the verb and noun a repair's wording counts it under.
_KINDS = {
    "replace": ("rewrote", "span"),
    "insert": ("inserted", "block"),
    "delete": ("deleted", "span"),
    "move": ("moved", "block"),
    "append": ("appended to", "file"),
    "create": ("created", "file"),
    "remove": ("removed", "file"),
    "rename": ("renamed", "path"),
    "import": ("added", "import"),
}


def grouped(cursor: DuckDBPyConnection, rule: Rule) -> defaultdict[object, list[Edit]]:
    """`rule`'s fix as edits grouped by the finding they repair.

    A rule fixing in Python repairs each finding row; a SQL fix states its edit rows, a row with
    no `finding_key` grouped alone. A row naming a structural operation that refuses withholds
    its whole group.
    """
    query = rule.edits("finding_rows")
    result = (
        cursor.execute("SELECT * FROM finding_rows ORDER BY finding_key")
        if query is None
        else cursor.execute(query.sql, query.parameters)
    )
    columns = [column for column, *_ in result.description]
    grouped: defaultdict[object, list[Edit]] = defaultdict(list)
    refused: set[object] = set()
    for number, row in enumerate(result.fetchall()):
        record = dict(zip(columns, row, strict=True))
        if query is None:
            grouped[record.pop("finding_key")].extend(rule.repair(Hit(**record)) or [])
        elif (made := _edits(record)) is None:
            refused.add(record.get("finding_key", ("row", number)))
        else:
            grouped[record.get("finding_key", ("row", number))].extend(made)
    for key in refused:
        grouped.pop(key, None)
    return grouped


def placed(rule: Rule, group: list[Edit], place: tuple[str, int, str | None] | None) -> dict:
    """The path, line and change text of a repair.

    That is the finding's own place and the rule's `change` template filled from its row. A group
    of no finding (a fix whose edit rows carry no `finding_key`) is placed at its first edit, and a
    rule without a template says what the edits did.
    """
    path, line, change = place or (group[0].file(), 1, None)
    return {"path": path, "line": line, "change": change or _described(rule, group)}


def _described(rule: Rule, edits: Iterable[Edit]) -> str:
    """What a fix without a `change` template did, in words.

    It names the kinds of edit the fix made and the rule by its name.
    """
    kinds = Counter(_KINDS[edit.kind] for edit in edits)
    made = ", ".join(f"{verb} {n} {noun}{'s' * (n > 1)}" for (verb, noun), n in kinds.items())
    return f"{made} ({snake(type(rule).__name__).replace('_', ' ')})"


def _edits(row: Mapping) -> list[Edit] | None:
    """The edits one row of a fix query states, None when its operation refuses.

    Each is a file operation (a new file, text appended, a removal, a rename to `target`), a
    structural operation (`operations`), a replaced span or a needed import.
    """
    if (operation := row.get("operation")) in _FILED:
        return [_operated(operation, row)]
    if operation is not None:
        return Operation.rendered(row)
    made: list[Edit] = []
    if row.get("text") is not None:
        line, column = row["start_line"], row["start_column"]
        span = Span(
            path=row["path"],
            start_line=line,
            start_column=column,
            end_line=row.get("end_line", line),
            end_column=row.get("end_column", column),
        )
        made.append(Replace(span=span, text=row["text"]))
    if row.get("import"):
        made.append(Import(path=row["path"], statement=row["import"]))
    return made


def _operated(operation: str, row: Mapping) -> Edit:
    """The file edit an `operation` row states: a new file, text appended, a removal, a rename."""
    match operation:
        case "append":
            return Append(path=row["path"], text=row["text"])
        case "create":
            return Create(path=row["path"], text=row["text"])
        case "remove":
            return Remove(path=row["path"])
        case "rename":
            return Rename(path=row["path"], target=row["target"])
    raise ValueError(f"a fix row states an unknown operation {operation!r}")
