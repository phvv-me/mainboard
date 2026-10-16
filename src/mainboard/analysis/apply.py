"""Applying fixes: edits become byte ranges in a plan, and plans are written into mirrors.

Span columns count UTF-8 bytes, as the facts do, and each file keeps its own line ending: a plan
works on bytes and writes CRLF into a file that uses it. A plan takes one finding's edits only
when they overlap none it already holds, so one pass applies every non-overlapping fix and the rest
wait.
A mirror is a hard-linked copy of the project's files that plans are written into as new files, so
fixes are tried and proven (fixing.py) without touching the project until the end.
"""

import ast
import bisect
import contextlib
import os
import re
import shutil
import tempfile
import tomllib
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

from patos import FrozenModel

from .edits import Delete, Edit, Import, Insert, Move, Remove, Rename, Replace
from .facts import listed

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .findings import Span

# A finding's place and rule: path, line, byte column, rule id.
type Spot = tuple[str, int, int, str]


class TextEdit(FrozenModel):
    """Replace bytes `[start, end)` of one file with `text`."""

    path: str
    start: int
    end: int
    text: bytes


class Source:
    """One file's bytes with line offsets, turning spans into byte ranges."""

    def __init__(self, data: bytes) -> None:
        self.data = data
        self.newline = b"\r\n" if b"\r\n" in data else b"\n"
        self.starts = [0, *(match.end() for match in re.finditer(b"\n", data))]
        if self.starts[-1] != len(data):
            self.starts.append(len(data))

    def offset(self, line: int, column: int) -> int:
        return self.starts[line - 1] + column

    def line(self, offset: int) -> int:
        """The line holding byte `offset`."""
        return bisect.bisect_right(self.starts, offset)

    def lines(self, span: Span) -> tuple[int, int]:
        """Whole lines `span` covers, line ending included: what deleting or moving it removes."""
        return self.starts[span.start_line - 1], self.starts[(span.end_line or span.start_line)]

    def indent(self, line: int) -> bytes:
        current = self.data[self.starts[line - 1] : self.starts[line]]
        return current[: len(current) - len(current.lstrip(b" \t"))]

    def encode(self, text: str) -> bytes:
        """`text` as this file's bytes, in its line ending."""
        return text.replace("\r\n", "\n").encode().replace(b"\n", self.newline)


class Plan:
    """The fixes of one pass over a project, expanded into byte edits, renames and removals."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.sources: dict[str, Source] = {}
        self.text: list[TextEdit] = []
        self.renamed: dict[str, str] = {}
        self.removed: list[str] = []
        self.imports: defaultdict[str, set[str]] = defaultdict(set)
        self.deferred = 0
        self._by_path: dict[str, list[TextEdit]] | None = None
        self._written: dict[str, bytes] | None = None

    def source(self, path: str) -> Source:
        if path not in self.sources:
            self.sources[path] = Source((self.root / path).read_bytes())
        return self.sources[path]

    def add(self, edits: Sequence[Edit]) -> bool:
        """Take one finding's edits unless they collide with ones already taken (or a rename
        would land on a path that exists); return whether."""
        expanded = [piece for edit in edits for piece in self._expand(edit)]
        removals = {edit.path for edit in edits if isinstance(edit, Remove)}
        renames = {edit.path: edit.target for edit in edits if isinstance(edit, Rename)}
        taken = {*self.removed, *self.renamed, *self.renamed.values()}
        if (
            ({*removals, *renames, *renames.values()} & taken)
            or any((self.root / target).exists() for target in renames.values())
            or any(_collide(edit, other) for edit in expanded for other in self.text)
        ):
            self.deferred += 1
            return False
        self.text.extend(expanded)
        self.removed.extend(removals)
        self.renamed.update(renames)
        self._by_path = self._written = None
        for edit in edits:
            if isinstance(edit, Import):
                self.imports[edit.path].add(edit.statement)
        return True

    def _expand(self, edit: Edit) -> list[TextEdit]:
        match edit:
            case Replace(span=span, text=text):
                source = self.source(span.path)
                start = source.offset(span.start_line, span.start_column)
                end = source.offset(span.end_line or span.start_line, span.end_column or 0)
                return [TextEdit(path=span.path, start=start, end=end, text=source.encode(text))]
            case Delete(span=span):
                start, end = self.source(span.path).lines(span)
                return [TextEdit(path=span.path, start=start, end=end, text=b"")]
            case Insert(text=text, before=before) as insert:
                anchor = insert.anchor()
                source = self.source(anchor.path)
                at = source.starts[anchor.start_line - 1] if before else source.lines(anchor)[1]
                indent = source.indent(anchor.start_line)
                lines = b"".join(
                    indent + source.encode(line) + source.newline for line in text.splitlines()
                )
                return [TextEdit(path=anchor.path, start=at, end=at, text=lines)]
            case Move(span=span, before=before, after=after):
                source = self.source(span.path)
                start, end = source.lines(span)
                width = len(source.indent(span.start_line))
                moved = "".join(
                    line[width:].decode() + "\n" for line in source.data[start:end].splitlines()
                )
                return [
                    TextEdit(path=span.path, start=start, end=end, text=b""),
                    *self._expand(Insert(text=moved, before=before, after=after)),
                ]
            case Remove() | Import() | Rename():
                return []
        raise TypeError(f"no expansion for edit kind {edit.kind}")

    def written(self) -> dict[str, bytes]:
        """Every edited file's new bytes."""
        if self._written is None:
            self._written = {}
            for path, edits in self._edits().items():
                data, pieces, at = self.source(path).data, [], 0
                for edit in edits:
                    pieces += [data[at : edit.start], edit.text]
                    at = edit.end
                self._written[path] = b"".join([*pieces, data[at:]])
        return self._written

    def touched(self) -> set[str]:
        """The project paths the plan changes (files, or directories it renames or removes), as
        they are before it."""
        return {*self._edits(), *self.removed, *self.renamed}

    def moved(self, path: str) -> str:
        """Where `path` stands after the plan's renames (of it or of a directory holding it)."""
        for old, new in self.renamed.items():
            if path == old or path.startswith(f"{old}/"):
                return new + path[len(old) :]
        return path

    def relocate(self, findings: Counter[Spot]) -> Counter[Spot]:
        """Each finding where the plan's edits put its start: a byte the edits kept moves with
        them, one they rewrote lands on its edit's start. A removed file's findings stay where
        they were, so nothing after the plan matches them."""
        edits = self._edits()
        after = {path: Source(data) for path, data in self.written().items()}
        moved: Counter[Spot] = Counter()
        for (path, line, column, rule), count in findings.items():
            if path in edits:
                source = self.source(path)
                at = source.starts[min(line, len(source.starts)) - 1] + column
                shift = 0
                for edit in edits[path]:
                    if edit.end <= at and (edit.start < at or edit.start == edit.end):
                        shift += len(edit.text) - (edit.end - edit.start)
                    elif edit.start <= at:
                        at = edit.start
                        break
                    else:
                        break
                line = after[path].line(at + shift)
                column = at + shift - after[path].starts[line - 1]
            moved[self.moved(path), line, column, rule] += count
        return moved

    def _edits(self) -> dict[str, list[TextEdit]]:
        """Each edited file's edits, its imports included, in file order."""
        if self._by_path is None:
            by_path = defaultdict(list)
            for edit in [*self.text, *self._imported()]:
                by_path[edit.path].append(edit)
            self._by_path = {
                path: sorted(edits, key=lambda e: (e.start, e.end))
                for path, edits in by_path.items()
            }
        return self._by_path

    def _imported(self) -> list[TextEdit]:
        """One insertion per file of the names its fixes import that its top-level imports do
        not already bind, after the last of them (else after its docstring); names from one
        module share one `from` statement. A statement that is no import (a module-level
        `logger = ...`) follows them after a blank line."""
        made = []
        for path, statements in self.imports.items():
            source = self.source(path)
            body = ast.parse(source.data).body
            bound = {binding for node in body for binding in _bindings(node)}
            wanted: defaultdict[str | None, list[str]] = defaultdict(list)
            definitions = []
            for statement in sorted(statements):
                node = ast.parse(statement).body[0]
                if not isinstance(node, ast.Import | ast.ImportFrom):
                    definitions.append(statement)
                for module, name in _bindings(node):
                    if (module, name) not in bound and name not in wanted[module]:
                        wanted[module].append(name)
            lines = [
                *(f"import {name}" for name in wanted.pop(None, [])),
                *(f"from {module} import {', '.join(names)}" for module, names in wanted.items()),
                *(["", *definitions] if definitions else []),
            ]
            if lines:
                at = source.starts[_import_line(body)]
                text = b"".join(source.encode(line) + source.newline for line in lines)
                made.append(TextEdit(path=path, start=at, end=at, text=text))
        return made


def _bindings(node: ast.stmt) -> list[tuple[str | None, str]]:
    """What an import statement binds: `(module, "name [as alias]")` for each name it takes
    from a module, `(None, "module [as alias]")` for each module it imports."""
    match node:
        case ast.ImportFrom(module=module, names=names, level=level):
            source = "." * level + (module or "")
            return [(source, _spelled(alias)) for alias in names]
        case ast.Import(names=names):
            return [(None, _spelled(alias)) for alias in names]
    return []


def _spelled(alias: ast.alias) -> str:
    return f"{alias.name} as {alias.asname}" if alias.asname else alias.name


def _import_line(body: list[ast.stmt]) -> int:
    """The zero-based line a new top-level import goes on."""
    imports = [node for node in body if isinstance(node, ast.Import | ast.ImportFrom)]
    if imports:
        return imports[-1].end_lineno or 0
    match body:
        case [ast.Expr(value=ast.Constant(value=str())) as docstring, *_]:
            return docstring.end_lineno or 0
    return 0


def _collide(one: TextEdit, other: TextEdit) -> bool:
    """Two edits of one file overlap, or insert at one point (their order would be a guess)."""
    if one.path != other.path:
        return False
    return (one.start < other.end and other.start < one.end) or (one.start == other.start)


class Mirror:
    """A hard-linked copy of a project's files (those git lists), changed only by writing new
    files, never through a link, so the files it was copied from stay as they are. The Ruff
    configurations the project extends (`extend = "../../../pyproject.toml"`) are linked at the
    same relative places, so the tools run in the copy read the configuration they read in
    place."""

    def __init__(self, source: Path) -> None:
        source = source.resolve()
        extended = _extended(source)
        anchor = Path(os.path.commonpath([source, *(file.parent for file in extended)]))
        if anchor == source:
            anchor = source.parent
        self.top = Path(tempfile.mkdtemp(prefix="mb-fix-"))
        self.path = self.top / source.relative_to(anchor)
        for name in listed(source):
            _link(source / name, self.path / name)
        for file in extended:
            _link(file, self.top / file.relative_to(anchor))
        self._beside(anchor, source)

    def _beside(self, anchor: Path, source: Path) -> None:
        """Link what lies beside the project and its parents up to `anchor`, as symbolic links to
        the real trees: type-checker settings reach them as `..` or `../../stubs`."""
        folder = anchor
        for part in source.relative_to(anchor).parts:
            for entry in folder.iterdir():
                target = self.top / entry.relative_to(anchor)
                if entry.name not in {part, ".git"} and not target.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.symlink_to(entry)
            folder /= part

    def apply(self, plan: Plan) -> set[str]:
        """Write `plan`; return the files it changed, by their names before and after it."""
        changed = set()
        for path, data in plan.written().items():
            target = self.path / path
            target.unlink()
            target.write_bytes(data)
            changed.add(path)
        for old, new in plan.renamed.items():
            changed |= {*self._files(old), *(plan.moved(name) for name in self._files(old))}
            (self.path / new).parent.mkdir(parents=True, exist_ok=True)
            (self.path / old).rename(self.path / new)
        for path in plan.removed:
            changed |= self._files(path)
            target = self.path / path
            shutil.rmtree(target) if target.is_dir() else target.unlink()
        return changed

    def adopt(self, source: Path, names: set[str]) -> None:
        """Make the files `names` here what they are under `source`, present or not."""
        for name in names:
            (self.path / name).unlink(missing_ok=True)
            if (source / name).exists() or (source / name).is_symlink():
                _link(source / name, self.path / name)

    def publish(self, root: Path, names: set[str]) -> None:
        """Write the files `names` into the project at `root` as they are here, removing those
        gone and the directories that leaves empty."""
        for name in sorted(names):
            source, target = self.path / name, root / name
            if source.is_symlink():
                target.unlink(missing_ok=True)
                target.symlink_to(os.readlink(source))
            elif source.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(source.read_bytes())
            else:
                target.unlink(missing_ok=True)
                for parent in PurePosixPath(name).parents:
                    with contextlib.suppress(OSError):
                        (root / parent).rmdir()

    def close(self) -> None:
        shutil.rmtree(self.top)

    def _files(self, path: str) -> set[str]:
        """The files at `path`: itself, or every file under it."""
        target = self.path / path
        if not target.is_dir():
            return {path}
        return {
            str(file.relative_to(self.path)) for file in target.rglob("*") if not file.is_dir()
        }


def _extended(project: Path) -> list[Path]:
    """The configuration files the Ruff configuration of `project` extends, their own extensions
    too, outside `project`; a `pyproject.toml` that does not extend one has none."""
    found: list[Path] = []
    pending = [project / name for name in ("pyproject.toml", "ruff.toml", ".ruff.toml")]
    while pending:
        file = pending.pop()
        try:
            table = tomllib.loads(file.read_text())
        except FileNotFoundError, tomllib.TOMLDecodeError:
            continue
        target = table.get("extend") if file.name != "pyproject.toml" else None
        target = target or table.get("tool", {}).get("ruff", {}).get("extend")
        if target:
            parent = (file.parent / target).resolve()
            if parent not in found and not parent.is_relative_to(project):
                found.append(parent)
            pending.append(parent)
    return found


def _link(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.is_symlink():
        target.symlink_to(os.readlink(source))
    else:
        os.link(source, target)


def under(path: str, roots: set[str]) -> bool:
    """Whether `path` is one of `roots` or lies in one of them."""
    return path in roots or any(str(parent) in roots for parent in PurePosixPath(path).parents)
