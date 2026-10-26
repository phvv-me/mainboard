# Reshaping class hierarchies: a method the siblings repeat pulled up into their base.

import ast
from typing import TYPE_CHECKING

from ...edits import Edit, Replace
from ...findings import Span
from ..base import Operation
from .source import Module

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


class PullUp(Operation):
    """Pull a method the siblings write alike up into their base, over its abstract stub.

    The first copy takes the place of the base's `@abstractmethod` placeholder, keeping the stub's
    docstring when the copy has none, and every copy goes with the blank lines above it.
    """

    @staticmethod
    def _carried(
        lines: Sequence[str],
        *,
        copy: ast.FunctionDef | ast.AsyncFunctionDef,
        stub: ast.FunctionDef | ast.AsyncFunctionDef,
        first: int,
    ) -> str:
        """The copy's text from line `first`, the stub's docstring set in when it has none."""
        text = list(lines[first - 1 : copy.end_lineno])
        if ast.get_docstring(copy) is None and ast.get_docstring(stub) is not None:
            opening = copy.body[0].lineno - first
            text[opening:opening] = lines[stub.body[0].lineno - 1 : stub.body[0].end_lineno]
        return "".join(text)

    @staticmethod
    def _method(tree: ast.Module, *, first: int) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
        """The method whose definition, decorators included, opens on line `first`."""
        return next(
            (
                node
                for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
                and min([node.lineno, *(each.lineno for each in node.decorator_list)]) == first
            ),
            None,
        )

    @staticmethod
    def _stub(
        tree: ast.Module, *, base: str, name: str
    ) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
        """The abstract placeholder `base` declares for `name`, None when it holds more."""
        found = next(
            (
                node
                for owner in tree.body
                if isinstance(owner, ast.ClassDef) and owner.name == base
                for node in owner.body
                if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name == name
            ),
            None,
        )
        if found is None:
            return None
        abstract = any(
            ast.unparse(each).rpartition(".")[2] == "abstractmethod"
            for each in found.decorator_list
        )
        placeholder = all(
            isinstance(each, ast.Pass | ast.Raise)
            or isinstance(each, ast.Expr)
            and isinstance(each.value, ast.Constant)
            for each in found.body
        )
        return found if abstract and placeholder else None

    def edits(self, row: Mapping) -> list[Edit] | None:
        content, path = row["content"], row["path"]
        module, lines = Module(content), content.splitlines(keepends=True)
        copied = row["members"][0]
        stub = PullUp._stub(module.tree, base=row["base"], name=row["name"])
        copy = PullUp._method(module.tree, first=copied["first_line"])
        if stub is None or copy is None:
            return None
        start = min([stub.lineno, *(each.lineno for each in stub.decorator_list)])
        span = Span(path=path, start_line=start, end_line=(stub.end_lineno or start) + 1)
        text = PullUp._carried(lines, copy=copy, stub=stub, first=copied["first_line"])
        return [
            Replace(span=span, text=text),
            *(
                module.cleared(path, first=member["first_line"], last=member["last_line"])
                for member in row["members"]
            ),
        ]
