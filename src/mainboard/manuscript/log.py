# What a TeX log says went wrong: every error, unresolved reference and citation, label defined
# twice and overfull box, each pinned to the source file and line it came from.
#
# The log never says which file a line belongs to. It prints `(path` when TeX opens a file and
# `)` when it closes one, so a warning belongs to whatever tops that stack when it prints. A
# parenthesis opening no file, `(12.3pt too wide)` say, is pushed and popped as an anonymous
# entry so the stack stays balanced.

import re
from enum import StrEnum, auto
from pathlib import Path, PurePosixPath

from patos import FrozenModel

# TeX breaks every log line at this many characters, so a line exactly this long continues on
# the next one.
_WRAP = 79

# A parenthesis and the path it opens, when it opens one. Engines name a file with or without
# its extension (tectonic writes `(sections/30_method`), so a path is anything that starts the
# way a path can, which `(12.3pt too wide)` does not; a word like `(Font)` that looks like one
# is closed again on the same line, so it never holds the top of the stack past its own text.
_PARENS = re.compile(r"\((?P<path>(?:[A-Za-z]:[\\/]|[./~A-Za-z_])[^\s(){}]*)?|\)")

# The suffix cleveref gives the shadow copy of every label it writes.
_SHADOW = "@cref"

_ERROR = re.compile(r"^! (?P<detail>.+)")
_ERROR_LINE = re.compile(r"^l\.(?P<line>\d+)")
_REFERENCE = re.compile(
    r"LaTeX Warning: Reference [`'](?P<name>[^']+)' on page \d+ undefined on input line "
    r"(?P<line>\d+)"
)
_CITATION = re.compile(
    r"(?:LaTeX|Package natbib) Warning: Citation [`'](?P<name>[^']+)' on page \d+ undefined "
    r"on input line (?P<line>\d+)"
)
_LABEL = re.compile(r"LaTeX Warning: Label [`'](?P<name>[^']+)' multiply defined")
_OVERFULL = re.compile(
    r"^Overfull \\[hv]box \((?P<amount>[^)]+)\).*?at lines? (?P<line>\d+)(?:--(?P<last>\d+))?"
)


class Kind(StrEnum):
    """What a build problem is, the column a reader sorts the report by."""

    ERROR = auto()
    REFERENCE = auto()
    CITATION = auto()
    LABEL = auto()
    OVERFULL = auto()
    LIMIT = auto()


class Problem(FrozenModel):
    """One thing the build found wrong, and where to go and fix it.

    where: `file:line` inside the manuscript directory, empty when TeX named no place.
    detail: the name, the amount or the message the log gave.
    """

    kind: Kind
    where: str = ""
    detail: str


class TexLog:
    """One TeX log, read into the problems it reports.

    directory: the manuscript directory the engine ran in, which file paths are shown under.
    """

    def __init__(self, text: str, *, directory: Path) -> None:
        self.lines = _unwrapped(text)
        self.directory = directory
        self.stack: list[str | None] = []
        self.closed = ""

    def problems(self) -> list[Problem]:
        """Every problem the log reports, in the order it reports them, each named once."""
        found: list[Problem] = []
        pending: Problem | None = None
        for line in self.lines:
            if pending and (numbered := _ERROR_LINE.match(line)):
                found.append(
                    pending.model_copy(update={"where": f"{pending.where}:{numbered[1]}"})
                )
                pending = None
            elif error := _ERROR.match(line):
                found.extend([pending] if pending else [])
                pending = Problem(
                    kind=Kind.ERROR, where=self.current, detail=error["detail"].strip()
                )
            else:
                found.extend(self._warnings(line))
        found.extend([pending] if pending else [])
        return list(dict.fromkeys(found))

    @property
    def current(self) -> str:
        """The file TeX is reading right now, shown relative to the manuscript directory."""
        opened = next((entry for entry in reversed(self.stack) if entry is not None), "")
        return self._shown(opened)

    def _warnings(self, line: str) -> list[Problem]:
        """The warnings `line` carries, attributed to the file open where each one prints."""
        matched = [
            (kind, found)
            for kind, pattern in (
                (Kind.REFERENCE, _REFERENCE),
                (Kind.CITATION, _CITATION),
                (Kind.LABEL, _LABEL),
                (Kind.OVERFULL, _OVERFULL),
            )
            if (found := pattern.search(line))
        ]
        if not matched:
            self._follow(line)
            return []
        kind, found = matched[0]
        self._follow(line[: found.start()])
        where = f"{self._source(found)}:{found['line']}" if "line" in found.groupdict() else ""
        detail = found["amount"] if kind is Kind.OVERFULL else found["name"]
        self._follow(line[found.end() :])
        # cleveref shadows every label with an `@cref` twin, so one duplicate warns twice.
        if detail.endswith(_SHADOW):
            return []
        return [Problem(kind=kind, where=where, detail=detail)]

    def _source(self, found: re.Match[str]) -> str:
        """The file a warning's first line sits in.

        A paragraph that starts at the end of one input and is broken after TeX has opened the
        next one reports lines that run backwards, `359--1`, and its start belongs to the file
        that just closed rather than to the one now open.
        """
        last = found.groupdict().get("last")
        if last and int(last) < int(found["line"]) and self.closed:
            return self._shown(self.closed)
        return self.current

    def _follow(self, text: str) -> None:
        """Move the open-file stack across every parenthesis in `text`."""
        for token in _PARENS.finditer(text):
            if token[0] == ")":
                if self.stack and (left := self.stack.pop()):
                    self.closed = left
                continue
            self.stack.append(token["path"])

    def _shown(self, path: str) -> str:
        """`path` as a reader finds it: relative to the manuscript directory when inside it."""
        if not path:
            return ""
        candidate = Path(path)
        if candidate.is_absolute():
            try:
                return candidate.relative_to(self.directory).as_posix()
            except ValueError:
                return candidate.as_posix()
        shown = str(PurePosixPath(path.replace("\\", "/"))).removeprefix("./")
        # An engine that names a source without its extension still means the `.tex` beside it.
        if not PurePosixPath(shown).suffix and (self.directory / f"{shown}.tex").is_file():
            return f"{shown}.tex"
        return shown


def _unwrapped(text: str) -> list[str]:
    """`text`'s lines with TeX's hard wrap undone, so a path or a warning reads whole."""
    joined: list[str] = []
    carried = ""
    for line in text.splitlines():
        carried += line
        if len(line) == _WRAP:
            continue
        joined.append(carried)
        carried = ""
    if carried:
        joined.append(carried)
    return joined
