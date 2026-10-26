"""Facts from the checkers mcmr leaves to their own tools: Ruff's diagnostics, each with the edits
of its fix, and ty's type errors.

A tool runs once over the project and its output becomes one fact family: the view
`<family>_diagnostics` (one row per diagnostic, placed like every fact) and, for a tool that
fixes, `<family>_edits` (one row per edit of a diagnostic's fix, keyed by its `fact_id`). The tools
report one-based character columns; spans here count zero-based UTF-8 bytes like every other fact.
Tools run only when a rule's SQL names their views, and together when several are named.
"""

import ast
import re
import subprocess  # ruff:ignore[suspicious-subprocess-import]  reason=runs ruff and ty with fixed argv since=2026-10-13
import sys
import tomllib
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from packaging.utils import canonicalize_name
from patos import FrozenModel, FrozenOpenModel
from pydantic import TypeAdapter

from ..engine.facts import listed, snake

if TYPE_CHECKING:
    from ..engine.facts import Facts, Query

type Row = tuple[str | int | None, ...]
type Table = tuple[dict[str, str], list[Row]]
type Place = tuple[int, int]

# Characters of file arguments one ruff command carries, well inside every kernel's limit.
_BATCH = 24_000

_DIAGNOSTICS = {
    "fact_id": "VARCHAR",
    "path": "VARCHAR",
    "language": "VARCHAR",
    "start_line": "BIGINT",
    "start_column": "BIGINT",
    "end_line": "BIGINT",
    "end_column": "BIGINT",
    "code": "VARCHAR",
    "name": "VARCHAR",
    "message": "VARCHAR",
    "severity": "VARCHAR",
    "fix_applicability": "VARCHAR",
    "fix_message": "VARCHAR",
}
_RUFF_DIAGNOSTICS = _DIAGNOSTICS | {"project": "VARCHAR"}
_EDITS = {
    "fact_id": "VARCHAR",
    "edit_index": "BIGINT",
    "path": "VARCHAR",
    "start_line": "BIGINT",
    "start_column": "BIGINT",
    "end_line": "BIGINT",
    "end_column": "BIGINT",
    "text": "VARCHAR",
    "import": "VARCHAR",
}


class Point(FrozenOpenModel):
    """A tool's one-based (line, character column)."""

    row: int
    column: int


class RuffEdit(FrozenOpenModel):
    content: str
    location: Point
    end_location: Point


class RuffFix(FrozenOpenModel):
    applicability: str
    message: str | None = None
    edits: list[RuffEdit]


class RuffDiagnostic(FrozenOpenModel):
    """One entry of `ruff check --output-format json`. A rule new in a release has a name and no
    code, and a syntax error is named `invalid-syntax`."""

    filename: str
    cell: int | None = None
    code: str | None = None
    name: str
    message: str
    severity: str
    location: Point
    end_location: Point
    fix: RuffFix | None = None


class Sources:
    """The project's files, each read once, for the positions the tools report in one-based
    characters and facts hold as zero-based UTF-8 bytes."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.files: dict[str, list[bytes]] = {}

    def _lines(self, path: str) -> list[bytes]:
        if path not in self.files:
            self.files[path] = (self.root / path).read_bytes().split(b"\n")
        return self.files[path]

    def place(self, path: str, point: Point) -> Place:
        """`point` as (line, byte column)."""
        lines = self._lines(path)
        if point.row > len(lines) or lines[point.row - 1].isascii():
            return point.row, point.column - 1
        before = lines[point.row - 1].decode(errors="replace")[: point.column - 1]
        return point.row, len(before.encode()) + point.column - 1 - len(before)

    def text(self, path: str, start: Place, *, end: Place) -> str:
        """The text between two places of `path`."""
        lines = self._lines(path)
        (first, begin), (last, stop) = start, end
        if first > len(lines):
            return ""
        if first == last:
            return lines[first - 1][begin:stop].decode(errors="replace")
        tail = lines[last - 1][:stop] if last <= len(lines) else b""
        return b"\n".join([lines[first - 1][begin:], *lines[first : last - 1], tail]).decode(
            errors="replace"
        )


@cache
def project_name(folder: Path) -> str | None:
    """The canonical `[project] name` of the nearest `pyproject.toml` at or above `folder`."""
    try:
        name = tomllib.loads((folder / "pyproject.toml").read_text())["project"]["name"]
    except FileNotFoundError, KeyError, NotADirectoryError, tomllib.TOMLDecodeError:
        return None if folder == folder.parent else project_name(folder.parent)
    return canonicalize_name(name)


class Tool(FrozenModel, ABC):
    """An external checker run over a project, its findings one fact family named `family`.

    extended: the diagnostic codes to add to the project's own configuration, which the rules
    reading the tool's views ask for in their `$extended` parameter.
    """

    family: ClassVar[str]
    seconds: ClassVar[int] = 600
    defined: ClassVar[dict[str, type[Tool]]] = {}
    extended: tuple[str, ...] = ()

    @abstractmethod
    def run(self, root: Path) -> dict[str, Table]:
        """The tool's findings over `root` as relations, the identity relation first."""

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: str) -> None:
        if "family" in cls.__dict__:
            Tool.defined[cls.family] = cls

    @classmethod
    def read(cls, queries: Iterable[Query]) -> list[Tool]:
        """The tools whose views `queries` name, each asked for the codes of those queries."""
        queries = list(queries)
        tools = []
        for family, tool in cls.defined.items():
            reading = [
                query
                for query in queries
                if any(view.startswith(f"{snake(family)}_") for view in query.views())
            ]
            if reading:
                codes = {
                    str(code)
                    for q in reading
                    if isinstance(extended := q.parameters.get("extended", ()), Sequence)
                    for code in extended
                }
                tools.append(tool(extended=tuple(sorted(codes))))
        return tools

    @classmethod
    def register_all(cls, facts: Facts, root: Path, queries: Iterable[Query]) -> None:
        """Run the tools `queries` read together, then register what each found."""
        tools = cls.read(queries)
        with ThreadPoolExecutor(len(tools) or 1) as pool:
            ran = pool.map(lambda tool: tool.run(root), tools)
            for tool, relations in zip(tools, ran, strict=True):
                tool.register(facts, relations)

    def register(self, facts: Facts, relations: Mapping[str, Table]) -> None:
        """Register `relations` as this tool's family."""
        for relation, (types, rows) in relations.items():
            columns = [list(c) for c in zip(*rows, strict=True)] if rows else [[]] * len(types)
            select = ", ".join(
                f"unnest(${name}::{sql}[]) AS {name}" for name, sql in types.items()
            )
            facts.db.execute(
                f'CREATE TABLE "{self.family}/{relation}/rows" AS SELECT {select}',
                dict(zip(types, columns, strict=True)),
            )
        facts.adopt(
            self.family,
            {relation: f'SELECT * FROM "{self.family}/{relation}/rows"' for relation in relations},
        )
        for relation in relations:
            facts.db.execute(f'DROP TABLE "{self.family}/{relation}/rows"')


class Ruff(Tool):
    """Ruff run with the project's own configuration plus `--extend-select` of the `extended`
    codes, and the edits of its fixes. The project's `ignore`, exclusions, per-file ignores,
    `# noqa`, fixability, target version and line length apply as they do to its own Ruff run, over
    the files `listed` keeps. Only the `extended` codes the project's own run does not report are
    kept (ruff itself, the hook's step, reports the rest), told apart by a plain run of the
    project's configuration beside the extended one. `limits` state the ceilings the dropped mcmr
    rules held, which the project's configuration would otherwise leave at Ruff's defaults.

    A fix that needs an import carries it as an edit inserting the statement (or rewriting the
    one already there as itself). Such an edit becomes the row's `import`, which the engine adds
    once per file however many fixes ask; left as text, the fixes sharing one insertion point
    would be applied one per pass. An import inserted where the same fix writes other text stays
    text, since the engine places its imports after that text and the text may use the name.
    """

    family = "Ruff"
    report: ClassVar[TypeAdapter[list[RuffDiagnostic]]] = TypeAdapter(list[RuffDiagnostic])
    limits: ClassVar[dict[str, int]] = {"max-args": 5, "max-positional-args": 5, "max-branches": 4}

    @staticmethod
    def _batches(names: Sequence[str]) -> Iterator[list[str]]:
        """`names` in runs whose characters stay under `BATCH`."""
        batch: list[str] = []
        size = 0
        for name in names:
            if batch and size + len(name) + 1 > _BATCH:
                yield batch
                batch, size = [], 0
            batch.append(name)
            size += len(name) + 1
        if batch:
            yield batch

    @staticmethod
    def _imports(content: str) -> list[str]:
        """The import statements `content` consists of; none when it is anything else.

        `__future__` imports are left out too, since they must stay the first statement of their
        file.
        """
        if not content.lstrip().startswith(("import ", "from ")):
            return []
        try:
            body = ast.parse(content).body
        except SyntaxError:
            return []
        if all(
            isinstance(node, ast.Import | ast.ImportFrom)
            and not (isinstance(node, ast.ImportFrom) and node.module == "__future__")
            for node in body
        ):
            return [ast.unparse(node) for node in body]
        return []

    @staticmethod
    def _key(item: RuffDiagnostic) -> tuple[str, int, int, str | None]:
        return item.filename, item.location.row, item.location.column, item.code

    def _check(
        self, root: Path, files: list[str], flags: Sequence[str] = ()
    ) -> list[RuffDiagnostic]:
        """Ruff's diagnostics over `files`, run in batches under the command-line budget.

        The files are named so Ruff's exclusions still apply.
        """
        found: list[RuffDiagnostic] = []
        for batch in Ruff._batches(files):
            output = subprocess.run(
                ["ruff", "check", "--output-format", "json", "--exit-zero", "--no-cache"]
                + ["--force-exclude", *flags, "--", *batch],
                cwd=root,
                capture_output=True,
                check=True,
                timeout=self.seconds,
            ).stdout
            found += self.report.validate_json(output)
        return found

    def _diagnostic(
        self, fact_id: str, *, path: str, item: RuffDiagnostic, sources: Sources
    ) -> Row:
        fix = item.fix
        return (
            fact_id,
            path,
            "python",
            *sources.place(path, item.location),
            *sources.place(path, item.end_location),
            item.code or item.name,
            item.name,
            item.message,
            item.severity,
            fix.applicability if fix else None,
            fix.message if fix else None,
            project_name((sources.root / path).resolve().parent),
        )

    def _edits(
        self, fact_id: str, *, path: str, fix: RuffFix | None, sources: Sources
    ) -> list[Row]:
        placed = [
            (sources.place(path, e.location), sources.place(path, e.end_location), e.content)
            for e in (fix.edits if fix else ())
        ]
        statements = [
            Ruff._imports(text)
            if start == end or sources.text(path, start, end=end) == text
            else []
            for start, end, text in placed
        ]
        written = {
            start for (start, _, _), found in zip(placed, statements, strict=True) if not found
        }
        rows: list[Row] = []
        for (start, end, text), found in zip(placed, statements, strict=True):
            if found and start not in written:
                rows += [(fact_id, len(rows), path, *start, *end, None, s) for s in found]
            else:
                rows.append((fact_id, len(rows), path, *start, *end, text, None))
        return rows

    def _lane(self, root: Path, files: list[str]) -> list[RuffDiagnostic]:
        """The `extended` diagnostics that the project's own configuration does not report."""
        own = {self._key(item) for item in self._check(root, files)}
        flags = [
            "--extend-select",
            ",".join(self.extended),
            *(
                arg
                for name, value in self.limits.items()
                for arg in ("--config", f"lint.pylint.{name}={value}")
            ),
        ]
        return [
            item
            for item in self._check(root, files, flags)
            if item.code in self.extended and self._key(item) not in own
        ]

    def _rows(self, root: Path, lane: Sequence[RuffDiagnostic]) -> tuple[list[Row], list[Row]]:
        """The diagnostic and edit rows of the `lane` diagnostics that name a Python file."""
        sources, base = Sources(root), root.resolve()
        diagnostics: list[Row] = []
        edits: list[Row] = []
        for number, item in enumerate(lane):
            path = Path(item.filename).relative_to(base).as_posix()
            if item.cell is None and path.endswith((".py", ".pyi")):
                fact_id = f"ruff:{number}"
                diagnostics.append(
                    self._diagnostic(fact_id, path=path, item=item, sources=sources)
                )
                edits += self._edits(fact_id, path=path, fix=item.fix, sources=sources)
        return diagnostics, edits

    def run(self, root: Path) -> dict[str, Table]:
        files = [name for name in listed(root) if name.endswith((".py", ".pyi"))]
        diagnostics, edits = self._rows(root, self._lane(root, files))
        return {"diagnostics": (_RUFF_DIAGNOSTICS, diagnostics), "edits": (_EDITS, edits)}


class Ty(Tool):
    """ty's type diagnostics, resolving imports against the Python running this. Its concise
    format places a diagnostic at its start only."""

    family = "Ty"
    pattern: ClassVar[re.Pattern[str]] = re.compile(
        r"^(?P<path>.+?):(?P<line>\d+):(?P<column>\d+): (?P<severity>\w+)\[(?P<code>[\w-]+)\] "
        r"(?P<message>.*)$"
    )

    def run(self, root: Path) -> dict[str, Table]:
        output = subprocess.run(
            ["ty", "check", "--python", sys.executable, "--project", ".", "--exit-zero"]
            + ["--output-format", "concise", "."],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
            timeout=self.seconds,
        ).stdout
        sources = Sources(root)
        diagnostics: list[Row] = []
        for line in output.splitlines():
            if found := self.pattern.match(line):
                point = Point(row=int(found["line"]), column=int(found["column"]))
                diagnostics.append(
                    (
                        f"ty:{len(diagnostics)}",
                        found["path"],
                        "python",
                        *sources.place(found["path"], point),
                        None,
                        None,
                        found["code"],
                        found["code"],
                        found["message"],
                        found["severity"],
                        None,
                        None,
                    )
                )
        return {"diagnostics": (_DIAGNOSTICS, diagnostics)}
