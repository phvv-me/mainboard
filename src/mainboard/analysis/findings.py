"""What the engine reports and hands to fixes: findings placed by a span, and each whole row."""

import json
import os
import re
from collections import Counter
from typing import TYPE_CHECKING

from patos import FrozenModel
from pydantic import ConfigDict

from ..core.project import Project

if TYPE_CHECKING:
    from collections.abc import Collection
    from pathlib import Path

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


class Report(FrozenModel):
    """Findings by rule, printed for a reader who pays for every token.

    A rule's message is printed once as a template, its varying parts holes; each file then lists
    its lines, every line with the values that fill its message's holes, in order. Nothing of a
    message is lost, and the prose a rule repeats a thousand times is read once.
    """

    found: dict[str, list[tuple[str, int, int, str]]]

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
                template, values = Report._split(message)
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

    @staticmethod
    def _split(message: str) -> tuple[str, list[str]]:
        """`message` as its template and the values filling the template's holes."""
        values = [named or number for url, named, number in _SLOT.findall(message) if not url]
        return _SLOT.sub(lambda match: match[1] or _HOLE, message), values


class Seen:
    """The findings each file held when it was last checked, kept beside the project's state.

    A finding counts by its rule and message, so one whose message changed (a statement count, a
    name) counts as a new one. Every check records what it saw, whether or not it reports only
    what is new. A session (`MB_REFACTOR_SESSION`, an agent's Stop hook) keeps a record of its own,
    started from the project's, so sessions editing one tree at once never take each other's
    findings as seen.
    """

    @staticmethod
    def _loaded(path: Path) -> dict[str, dict[str, int]]:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError, json.JSONDecodeError:
            return {}

    def __init__(self, root: Path) -> None:
        folder = Project().out(root) / "refactor"
        session = Project().variable("REFACTOR_SESSION").read()
        self.path = folder / "sessions" / f"{session}.json" if session else folder / "seen.json"
        self.held = Seen._loaded(self.path) or Seen._loaded(folder / "seen.json")

    def fresh(self, report: Report, files: Collection[str]) -> Report:
        """The findings of `files` beyond what each held when last recorded, recording them now.

        The first check of a project records without reporting; after it, a file it never recorded
        is a new one, all of whose findings are new. Only the rules `report` ran are recorded, so a
        check of a few rules leaves what the others last saw.
        """
        now: dict[str, Counter[str]] = {path: Counter() for path in files}
        for rule, rows in report.found.items():
            for path, _, _, message in rows:
                now.setdefault(path, Counter())[f"{rule}\t{message}"] += 1
        grown = {
            path: {key for key, count in counts.items() if count > held.get(key, 0)}
            for path, counts in now.items()
            for held in [self.held.get(path, {})]
            if self.held
        }
        ran = set(report.found)
        self.held.update(
            {
                path: {
                    key: count
                    for key, count in self.held.get(path, {}).items()
                    if key.partition("\t")[0] not in ran
                }
                | dict(counts)
                for path, counts in now.items()
            }
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        staged = self.path.with_suffix(f".{os.getpid()}.tmp")
        staged.write_text(json.dumps(self.held, sort_keys=True))
        os.replace(staged, self.path)
        return Report(
            found={
                rule: [row for row in rows if f"{rule}\t{row[3]}" in grown.get(row[0], set())]
                for rule, rows in report.found.items()
            }
        )
