"""Applying fixes: edits become byte ranges in a plan, which a mirror (mirror.py) then writes.

Span columns count UTF-8 bytes, as the facts do, and each file keeps its own line ending: a plan
works on bytes and writes CRLF into a file that uses it. A plan takes one finding's edits only
when they overlap none it already holds, so one pass applies every non-overlapping fix and the rest
wait.
"""

import ast
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

from ...lint.sealed import Sealed
from ..edits import Append, Create, Delete, Edit, Import, Insert, Move, Remove, Rename, Replace
from .sources import Source, TextEdit

if TYPE_CHECKING:
    from collections.abc import Sequence, Set


# A finding's place and rule: path, line, byte column, rule id.
type Spot = tuple[str, int, int, str]


class Plan:
    """The fixes of one pass over a project, expanded into byte edits, renames and removals."""

    # How many findings' edits the pass put off, since they collided with edits it took.
    deferred = 0

    @staticmethod
    def _collide(one: TextEdit, *, other: TextEdit) -> bool:
        """Two edits of one file overlap, or insert at one point (their order would be a guess).

        Appends stack in the order they were taken, so two of them never collide.
        """
        if one.path != other.path or (one.stacks and other.stacks):
            return False
        return (one.start < other.end and other.start < one.end) or (one.start == other.start)

    @staticmethod
    def _import_line(body: Iterable[ast.stmt]) -> int:
        """The zero-based line a new top-level import goes on."""
        imports = [node for node in body if isinstance(node, ast.Import | ast.ImportFrom)]
        if imports:
            return imports[-1].end_lineno or 0
        match body:
            case [ast.Expr(value=ast.Constant(value=str())) as docstring, *_]:
                return docstring.end_lineno or 0
        return 0

    @staticmethod
    def _shifted(edits: Sequence[TextEdit], offset: int) -> int:
        """Where byte `offset` stands after `edits`, in file order.

        A byte the edits kept moves with them; one they rewrote lands on its edit's start.
        """
        shift = 0
        for edit in edits:
            if edit.start > offset:
                break
            if edit.end > offset:
                return edit.start + shift
            shift += len(edit.text) - (edit.end - edit.start)
        return offset + shift

    @staticmethod
    def _wanted(
        statements: Iterable[str], bound: Set[tuple[str | None, str]]
    ) -> tuple[defaultdict[str | None, list[str]], list[str]]:
        """The names `statements` import that the file does not bind yet, by module.

        Beside them come the statements that are no import at all (a module-level `logger = ...`).
        """
        wanted: defaultdict[str | None, list[str]] = defaultdict(list)
        definitions = []
        for statement in sorted(statements):
            node = ast.parse(statement).body[0]
            if not isinstance(node, ast.Import | ast.ImportFrom):
                definitions.append(statement)
            for module, name in _bindings(node):
                if (module, name) not in bound and name not in wanted[module]:
                    wanted[module].append(name)
        return wanted, definitions

    def __init__(self, root: Path) -> None:
        self.root = root
        self.sealed = Sealed(root)
        self.sources: dict[str, Source] = {}
        self.text: list[TextEdit] = []
        self.renamed: dict[str, str] = {}
        self.removed: list[str] = []
        self.created: set[str] = set()
        self.imports: defaultdict[str, set[str]] = defaultdict(set)
        self._by_path: dict[str, list[TextEdit]] | None = None
        self._written: dict[str, bytes] | None = None

    def _anchored(self, edit: Edit) -> list[TextEdit]:
        """The byte edits of an edit placed by an anchor, none for one that changes no text."""
        match edit:
            case Insert(text=text, before=before) as insert:
                anchor = insert.anchor()
                source = self.source(anchor.path)
                start = source.starts[anchor.start_line - 1] if before else source.lines(anchor)[1]
                indent = source.indent(anchor.start_line)
                lines = b"".join(
                    indent + source.encode(line) + source.newline for line in text.splitlines()
                )
                return [TextEdit(path=anchor.path, start=start, end=start, text=lines)]
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
            case Create(path=path, text=text):
                return [TextEdit(path=path, start=0, end=0, text=text.encode())]
            case Append(path=path, text=text):
                source = self.source(path)
                end = len(source.data)
                parted = source.data and not source.data.endswith(b"\n")
                data = (source.newline if parted else b"") + source.encode(text)
                return [TextEdit(path=path, start=end, end=end, text=data, stacks=True)]
        raise TypeError(f"no expansion for edit kind {edit.kind}")

    def _collides(self, edits: Sequence[Edit], expanded: Sequence[TextEdit]) -> bool:
        """Whether one finding's edits clash with what the plan already holds.

        A path it removes or renames is one the plan already moves, a rename lands on a path that
        exists, or a byte edit overlaps one already taken.
        """
        moving = {edit.path for edit in edits if isinstance(edit, Remove | Rename)}
        targets = {edit.target for edit in edits if isinstance(edit, Rename)}
        made = {edit.path for edit in edits if isinstance(edit, Create)}
        taken = {*self.removed, *self.renamed, *self.renamed.values(), *self.created}
        return (
            bool((moving | targets | made) & taken)
            or any((self.root / target).exists() for target in targets | made)
            or any(Plan._collide(edit, other=other) for edit in expanded for other in self.text)
        )

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

    def _expand(self, edit: Edit) -> list[TextEdit]:
        match edit:
            case Replace(span=span, text=text):
                source = self.source(span.path)
                start = source.offset(span.start_line, column=span.start_column)
                end = source.offset(span.end_line or span.start_line, column=span.end_column or 0)
                return [TextEdit(path=span.path, start=start, end=end, text=source.encode(text))]
            case Delete(span=span):
                start, end = self.source(span.path).lines(span)
                return [TextEdit(path=span.path, start=start, end=end, text=b"")]
        return self._anchored(edit)

    def _import(self, path: str, statements: Iterable[str]) -> TextEdit | None:
        """The one insertion into `path` of what its fixes import and it does not yet bind.

        It goes after the last top-level import (else after the docstring), and names from one
        module share one `from` statement. A statement that is no import follows them after a
        blank line.
        """
        source = self.source(path)
        body = ast.parse(source.data).body
        bound = {binding for node in body for binding in _bindings(node)}
        wanted, definitions = Plan._wanted(statements, bound)
        lines = [
            *(f"import {name}" for name in wanted.pop(None, [])),
            *(f"from {module} import {', '.join(names)}" for module, names in wanted.items()),
            *(["", *definitions] if definitions else []),
        ]
        if not lines:
            return None
        start = source.starts[Plan._import_line(body)]
        text = b"".join(source.encode(line) + source.newline for line in lines)
        return TextEdit(path=path, start=start, end=start, text=text)

    def _imported(self) -> list[TextEdit]:
        """One insertion per file of the names its fixes import and it does not yet bind."""
        return [
            edit
            for path, statements in self.imports.items()
            if (edit := self._import(path, statements)) is not None
        ]

    def _relocated(self, path: str, *, line: int, column: int, after: Source) -> tuple[int, int]:
        """Where a finding at `line` and `column` of `path` stands in `after`, its new bytes."""
        source = self.source(path)
        offset = source.starts[min(line, len(source.starts)) - 1] + column
        shifted = Plan._shifted(self._edits().get(path, []), offset)
        moved_line = after.line(shifted)
        return moved_line, shifted - after.starts[moved_line - 1]

    def _spliced(self, path: str, edits: Sequence[TextEdit]) -> bytes:
        """`path`'s bytes with `edits`, in file order, applied."""
        data, pieces, kept = self.source(path).data, [], 0
        for edit in edits:
            pieces += [data[kept : edit.start], edit.text]
            kept = edit.end
        return b"".join([*pieces, data[kept:]])

    def add(self, edits: Sequence[Edit]) -> bool:
        """Take one finding's edits unless they collide; answer whether they were taken.

        They collide with edits already taken, with a rename landing on a path that exists, or with
        any change to a file a registration has sealed.
        """
        sealed = any(self.sealed.holds(self.root / edit.file()) for edit in edits)
        expanded = [] if sealed else [piece for edit in edits for piece in self._expand(edit)]
        if sealed or self._collides(edits, expanded):
            self.deferred += 1
            return False
        self.text.extend(expanded)
        self.removed.extend(edit.path for edit in edits if isinstance(edit, Remove))
        self.created.update(edit.path for edit in edits if isinstance(edit, Create))
        self.renamed.update({edit.path: edit.target for edit in edits if isinstance(edit, Rename)})
        self._by_path = self._written = None
        for edit in edits:
            if isinstance(edit, Import):
                self.imports[edit.path].add(edit.statement)
        return True

    def moved(self, path: str) -> str:
        """Where `path` stands after the plan's renames (of it or of a directory holding it)."""
        for old, new in self.renamed.items():
            if path == old or path.startswith(f"{old}/"):
                return new + path[len(old) :]
        return path

    def relocate(self, findings: Counter[Spot]) -> Counter[Spot]:
        """Each finding where the plan's edits put its start.

        A byte the edits kept moves with them, one they rewrote lands on its edit's start. A
        removed file's findings stay where they were, so nothing after the plan matches them.
        """
        after = {path: Source(data) for path, data in self.written().items()}
        moved: Counter[Spot] = Counter()
        for (path, line, column, rule), count in findings.items():
            if path in after:
                line, column = self._relocated(path, line=line, column=column, after=after[path])
            moved[self.moved(path), line, column, rule] += count
        return moved

    def source(self, path: str) -> Source:
        """`path`'s bytes before the plan, empty for a file it creates or appends into being."""
        if path not in self.sources:
            try:
                data = b"" if path in self.created else (self.root / path).read_bytes()
            except FileNotFoundError:
                data = b""
            self.sources[path] = Source(data)
        return self.sources[path]

    def touched(self) -> set[str]:
        """The project paths the plan changes, as they are before it.

        Those are files, and the directories it renames or removes.
        """
        return {*self._edits(), *self.removed, *self.renamed}

    def written(self) -> dict[str, bytes]:
        """Every edited file's new bytes."""
        if self._written is None:
            self._written = {
                path: self._spliced(path, edits) for path, edits in self._edits().items()
            }
        return self._written


def _bindings(node: ast.stmt) -> list[tuple[str | None, str]]:
    """What an import statement binds, one pair per name or module it takes.

    Each name taken from a module is `(module, "name [as alias]")`, and each module imported is
    `(None, "module [as alias]")`.
    """
    match node:
        case ast.ImportFrom(module=module, names=names, level=level):
            source = "." * level + (module or "")
            return [(source, _spelled(alias)) for alias in names]
        case ast.Import(names=names):
            return [(None, _spelled(alias)) for alias in names]
    return []


def _spelled(alias: ast.alias) -> str:
    return f"{alias.name} as {alias.asname}" if alias.asname else alias.name
