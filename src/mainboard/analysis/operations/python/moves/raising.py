# Moving a constant up to where it can first stand.

import ast
from typing import TYPE_CHECKING

from ....edits import Edit, Insert
from ....findings import Span
from ...base import Operation
from ..source import Module

if TYPE_CHECKING:
    from collections.abc import Mapping


class RaiseConstants(Operation):
    """Move each constant a module declares late up to just below what its value reads.

    That is the last import or the last statement binding a name the value reads, whichever comes
    later, with the comments above the constant going along. A constant already there stays.
    """

    @staticmethod
    def _gap(anchor: ast.stmt) -> str:
        """The blank lines a formatter parts a constant from `anchor` with."""
        if isinstance(anchor, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            return "\n\n"
        return "\n" if isinstance(anchor, ast.Import | ast.ImportFrom) else ""

    @staticmethod
    def _raised(module: Module, *, path: str, name: str) -> list[Edit] | None:
        """The edits putting `name`'s block just below what it reads, None when it is there.

        The block leaves with the blank lines above it and arrives parted from what it follows as
        a formatter parts them: a line after an import, two after a function or a class.
        """
        body = module.tree.body
        index = next((i for i, node in enumerate(body) if Module.target(node) == name), 0)
        block = module.block(name, literal=False)
        reads = {each.id for each in ast.walk(body[index]) if isinstance(each, ast.Name)}
        needed = [
            i
            for i, node in enumerate(body[:index])
            if isinstance(node, ast.Import | ast.ImportFrom) or Module.binds(node) & reads
        ]
        if block is None or not index or not needed or needed[-1] == index - 1:
            return None
        anchor = body[needed[-1]]
        below = Span(path=path, start_line=anchor.lineno, end_line=anchor.end_lineno)
        return [
            module.cleared(path, first=block.first, last=block.last),
            Insert(text=RaiseConstants._gap(anchor) + block.text, after=below),
        ]

    def edits(self, row: Mapping) -> list[Edit] | None:
        module = Module(row["content"])
        moves = [
            RaiseConstants._raised(module, path=row["path"], name=name) for name in row["names"]
        ]
        return [edit for move in moves if move is not None for edit in move] or None
