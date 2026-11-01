# Renaming a binding and every reference to it: a function's local through the scopes symtable
# states, a module's name through the files that import it.

from typing import TYPE_CHECKING

from pydantic import ValidationError

from ....edits import Edit, Replace
from ....findings import Span
from ...base import Operation
from ..imports import Statement
from .rows import Importer, Reading, RenameRow
from .scopes import Scopes

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


class RewriteReferences(Operation):
    """Rename a binding and every reference to it.

    A function's local is renamed in the function and in the scopes inside it that read it. A
    module's name is renamed where it is defined and in `__all__`, and in each file importing it:
    the `from ... import`, the reads that follow it and each `module.name` read. The rename is
    refused when the new name is already a name in a scope it reaches, when a token spelling the
    old name is neither rewritten nor placed by the syntax tree as another name (a string,
    `global`, `except ... as`, an f-string field printing its own source) in the files it
    rewrites or in any other Python file importing the module or spelling `module.name`, or when
    the scopes cannot be told apart. Comments and strings are never rewritten.
    """

    @staticmethod
    def _unsafe(scopes: Scopes, row: RenameRow, spans: Sequence[Span], *, line: int) -> str | None:
        """Why renaming `spans` in the scope at `line` would change what it means, else None."""
        if scopes.reflective(line):
            return "the scope reads its names by their spelling (`locals`, `eval`)"
        if scopes.clash(line, row.to):
            return f"`{row.to}` is already a name in the scope or in a scope inside it"
        if not scopes.spelled(spans, row.name, line=line):
            return f"a token spells `{row.name}` where the syntax tree places no variable of it"
        return None

    def _accounted(self, scopes: Scopes, row: RenameRow, spans: Sequence[Span]) -> bool:
        """Whether `spans` and the syntax tree account for every spelling of the name in a file."""
        if scopes.spelled(spans, row.name):
            return True
        self.refused(
            f"`{scopes.path}` spells `{row.name}` where the syntax tree places no variable of it"
        )
        return False

    def _followed(self, scopes: Scopes, row: RenameRow) -> list[Span] | None:
        """The spans of the reads of a name an importer binds, which follow it to its new name."""
        if (reads := scopes.module_spans(row.name)) is None:
            self.refused(f"`{scopes.path}` binds `{row.name}` in scopes it cannot tell apart")
            return None
        if scopes.clash(0, row.to):
            self.refused(f"`{row.to}` is already a name in `{scopes.path}`")
            return None
        return reads

    def _function(self, row: RenameRow) -> list[Span] | None:
        """The spans renaming the local of the function opening on the row's line."""
        if (scopes := self._parsed(row.content, path=row.path)) is None:
            return None
        if (spans := scopes.local_spans(row.line or 0, row.name)) is None:
            self.refused(f"`{row.name}` is no plain local of the function")
            return None
        if why := RewriteReferences._unsafe(scopes, row, spans, line=row.line or 0):
            self.refused(why)
            return None
        return spans

    def _imports(
        self, scopes: Scopes, row: RenameRow, imports: Sequence[Importer]
    ) -> list[Span] | None:
        """The spans renaming the name in the `from ... import`s of one file and what they bind.

        The reads follow when a statement binds the name itself, with no `as`.
        """
        statements = [Statement(each.model_dump(), destination=row.module) for each in imports]
        renamed = [statement.renamed(row.name, new=row.to) for statement in statements]
        if None in renamed:
            self.refused(f"`{scopes.path}` does not import `{row.name}` as the row says")
            return None
        if not all(scopes.imported(each.start_line) for each in imports):
            self.refused(f"`{scopes.path}` imports `{row.name}` inside a function or class")
            return None
        spans = [edit.span for edits in renamed for edit in edits or []]
        if not any(statement.binds(row.name) for statement in statements):
            return spans
        reads = self._followed(scopes, row)
        return None if reads is None else [*spans, *reads]

    def _module(self, row: RenameRow) -> list[Span] | None:
        """The spans renaming the module's name where defined and in every file reading it."""
        if (scopes := self._parsed(row.content, path=row.path)) is None:
            return None
        if row.starred or (spans := scopes.module_spans(row.name)) is None:
            self.refused(f"`{row.name}` is no plain module-level binding, or is starred")
            return None
        if why := RewriteReferences._unsafe(scopes, row, spans, line=0):
            self.refused(why)
            return None
        paths = {each.path for each in (*row.importers, *row.reached, *row.dotted)}
        readers = [self._reader(row, path) for path in sorted(paths)]
        silent = [each for each in row.scanned if each.path not in {row.path, *paths}]
        if None in readers or not all(self._silent(row, each) for each in silent):
            return None
        return [*spans, *(s for each in readers for s in each or [])]

    def _parsed(self, content: str, *, path: str) -> Scopes | None:
        """The scopes of `content`, None when it is no Python."""
        try:
            return Scopes(content, path=path)
        except SyntaxError:
            self.refused(f"`{path}` does not parse")
            return None

    def _reader(self, row: RenameRow, path: str) -> list[Span] | None:
        """The spans renaming the name in the file at `path`, which reads it from the module."""
        imports = [each for each in row.importers if each.path == path]
        reaches = [each for each in (*row.reached, *row.dotted) if each.path == path]
        content = next(each.content for each in (*imports, *reaches))
        if (scopes := self._parsed(content, path=path)) is None:
            return None
        if (spans := self._imports(scopes, row, imports)) is None:
            return None
        spans += [each.tail(row.name) for each in reaches]
        return spans if self._accounted(scopes, row, spans) else None

    def _silent(self, row: RenameRow, file: Reading) -> bool:
        """Whether `file`, which the rename leaves as it is, spells the name nowhere."""
        scopes = self._parsed(file.content, path=file.path)
        return scopes is not None and self._accounted(scopes, row, [])

    def edits(self, row: Mapping) -> list[Edit] | None:
        try:
            asked = RenameRow(**row)
        except ValidationError as error:
            self.refused("the row is malformed: " + error.errors()[0]["msg"])
            return None
        spans = self._function(asked) if asked.scope == "function" else self._module(asked)
        distinct = {(s.path, s.start_line, s.start_column): s for s in spans or []}
        return (
            None if spans is None else [Replace(span=s, text=asked.to) for s in distinct.values()]
        )
