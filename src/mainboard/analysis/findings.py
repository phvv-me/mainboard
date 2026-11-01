"""What the engine reports and hands to fixes: findings placed by a span, and each whole row."""

import re
from typing import TYPE_CHECKING

from patos import FrozenModel
from pydantic import ConfigDict

if TYPE_CHECKING:
    from collections.abc import Collection

_PLACE = ("path", "start_line", "start_column")
_ENDS = ("end_line", "end_column")


class Span(FrozenModel):
    """A source range, lines one-indexed and columns zero-indexed, as the kernel reports them."""

    path: str
    start_line: int
    start_column: int = 0
    end_line: int | None = None
    end_column: int | None = None


class Finding(FrozenModel):
    """One finding, the unit compared, reported and fixed."""

    rule: str
    span: Span
    message: str


type Cell = str | int | float | bool | None


class Hit(FrozenModel):
    """One finding's whole row, handed to its rule's fix; columns read as `hit.name`."""

    model_config = ConfigDict(extra="allow")

    @property
    def span(self) -> Span:
        """Where the finding sits."""
        row = self.model_extra or {}
        return Span(**{key: row[key] for key in (*_PLACE, *_ENDS) if key in row})

    def __getattr__(self, column: str) -> Cell:
        try:
            return (self.model_extra or {})[column]
        except KeyError:
            raise AttributeError(f"this finding's row has no column {column!r}") from None

    def at(self, prefix: str) -> Span:
        """The span the row's `{prefix}_path`, `{prefix}_start_line`, ... columns hold."""
        row = self.model_extra or {}
        return Span(**{key: row[f"{prefix}_{key}"] for key in (*_PLACE, *_ENDS)})

    def records(self, column: str) -> list[dict[str, int | str]]:
        """The structs a list column holds (`members`, `calls`), empty when the row has none."""
        return (self.model_extra or {}).get(column) or []

    def sequence(self, column: str) -> list[Cell]:
        """The list a column holds, empty when the row has none."""
        return (self.model_extra or {}).get(column) or []


# What varies between two findings of one message: a backticked name, or a number standing alone.
_SLOT = re.compile(r"(https?://[^\s`]+)|`([^`]*)`|(?<![\w.])(\d+(?:\.\d+)?)(?![\w.])")
_HOLE = "_"
# A list of holes, which a finding's template keeps as one, so a list grown by a name is the same.
_LISTED = re.compile(r"_(?:(?:, |, and | and |\|)_)+")

# One finding as a report holds it: path, line (0 for none), column, message.
type Row = tuple[str, int, int, str]


class Report(FrozenModel):
    """Findings by rule, printed for a reader who pays for every token.

    A rule's message is printed once as a template, its varying parts holes; each file then lists
    its lines, every line with the values that fill its message's holes, in order. Nothing of a
    message is lost, and the prose a rule repeats a thousand times is read once.
    """

    found: dict[str, list[Row]]

    @property
    def counts(self) -> str:
        """One line per rule, the rules with most findings first, then the total."""
        ranked = sorted(self.found.items(), key=lambda item: (-len(item[1]), item[0]))
        lines = [f"{len(rows):5d} {rule}" for rule, rows in ranked if rows]
        return "\n".join([*lines, f"{self.total:5d} findings, {len(lines)} rules"])

    @property
    def grouped(self) -> str:
        """Each rule's template, then each file's lines and the values filling its holes."""
        sites: dict[tuple[str, str], dict[str, list[tuple[int, list[str]]]]] = {}
        for rule, rows in sorted(self.found.items()):
            for path, line, _, message in rows:
                template, values = _split(message)
                sites.setdefault((rule, template), {}).setdefault(path, []).append((line, values))
        lines: list[str] = []
        for (rule, template), files in sites.items():
            lines.append(f"{rule} {template}")
            for path, found in sorted(files.items()):
                cells = (
                    f"{line}[{'|'.join(values)}]" if values else f"{line}"
                    for line, values in sorted(found)
                )
                lines.append(f" {path} {' '.join(cells)}")
        return "\n".join(lines)

    @property
    def total(self) -> int:
        return sum(len(rows) for rows in self.found.values())

    def within(self, files: Collection[str]) -> Report:
        """The findings in `files`."""
        return Report(
            found={
                rule: [row for row in rows if row[0] in files] for rule, rows in self.found.items()
            }
        )


def quoted(message: str) -> list[str]:
    """The names `message` quotes in backticks, in order."""
    return [named for _, named, _ in _SLOT.findall(message) if named]


def shape(message: str) -> str:
    """`message`'s template with each list of holes as one hole; the report keeps every hole."""
    return _LISTED.sub(_HOLE, _split(message)[0])


def _split(message: str) -> tuple[str, list[str]]:
    """`message` as its template and the values filling the template's holes."""
    values = [named or number for url, named, number in _SLOT.findall(message) if not url]
    return _SLOT.sub(lambda match: match[1] or _HOLE, message), values
