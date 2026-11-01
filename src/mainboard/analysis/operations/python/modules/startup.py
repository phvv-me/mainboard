# Deferring a module's imports with PEP 810's `__lazy_modules__`: Python 3.15 loads a listed
# module's module-level import when a name the import binds is first read, 3.14 ignores the list.

import ast
from typing import TYPE_CHECKING

from ....edits import Edit, Replace
from ....findings import Span
from ...base import Operation
from ..imports import Place
from ..source import Module

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence


class LazyModules(Operation):
    """List modules in a file's `__lazy_modules__`, ahead of its imports.

    The list keeps the modules it listed that the file still imports and gains the row's
    `entries`, and stands before the first import, since 3.15 reads it as the imports after it run.
    """

    @staticmethod
    def _dropped(path: str, node: ast.stmt, lines: Sequence[str]) -> Span:
        """`node`'s whole lines and the blank line after them, which deleting it leaves over."""
        end = (node.end_lineno or node.lineno) + 1
        end += end <= len(lines) and not lines[end - 1].strip()
        return Span(path=path, start_line=node.lineno, end_line=end)

    @staticmethod
    def _held(tree: ast.Module) -> list[ast.Assign]:
        """The module-level assignments of `__lazy_modules__` in `tree`."""
        return [
            node
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "__lazy_modules__" for t in node.targets)
        ]

    @staticmethod
    def _imported(tree: ast.Module, place: Place) -> set[str]:
        """Every name a module-level import of `tree` is deferred by (see `ImportsModuleEdges`)."""
        named = set()
        for node in tree.body:
            match node:
                case ast.Import(names=names):
                    named |= {alias.name for alias in names}
                case ast.ImportFrom():
                    named.add(place.resolved(node))
        return named

    @staticmethod
    def _listed(held: Iterable[ast.Assign]) -> set[str]:
        """The modules the assignments `held` list."""
        return {
            item.value
            for node in held
            for item in ast.walk(node.value)
            if isinstance(item, ast.Constant) and isinstance(item.value, str)
        }

    @staticmethod
    def _listing(entries: Iterable[str]) -> str:
        """`__lazy_modules__` holding `entries`, sorted, one per line."""
        return "__lazy_modules__ = [\n" + "".join(f'    "{e}",\n' for e in sorted(entries)) + "]"

    @staticmethod
    def _placed(
        path: str, tree: ast.Module, *, lines: Sequence[str], held: Sequence[ast.Assign], text: str
    ) -> list[Edit]:
        """The edits putting `text` where the file's list goes.

        It replaces the first of `held` when that stands before every import, else it goes before
        the first import and every one of `held` is deleted.
        """
        first = next(node for node in tree.body if isinstance(node, ast.Import | ast.ImportFrom))
        if held and held[0].lineno < first.lineno:
            return [Replace(span=LazyModules._span(path, held[0]), text=text)]
        start = Span(path=path, start_line=first.lineno, end_line=first.lineno)
        return [
            Replace(span=start, text=text + "\n\n"),
            *(Replace(span=LazyModules._dropped(path, node, lines), text="") for node in held),
        ]

    @staticmethod
    def _span(path: str, node: ast.stmt) -> Span:
        """Where `node` stands in the file at `path`."""
        return Span(
            path=path,
            start_line=node.lineno,
            start_column=node.col_offset,
            end_line=node.end_lineno,
            end_column=node.end_col_offset,
        )

    def edits(self, row: Mapping) -> list[Edit] | None:
        """The file's list with the row's `entries` added, `importer` being the file's module."""
        path, module = row["path"], Module(row["content"])
        held = LazyModules._held(module.tree)
        imported = LazyModules._imported(module.tree, Place(path, row["importer"]))
        text = LazyModules._listing((LazyModules._listed(held) & imported) | set(row["entries"]))
        return LazyModules._placed(
            path, module.tree, lines=module.content.splitlines(), held=held, text=text
        )
