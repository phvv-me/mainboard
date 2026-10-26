"""What a Python module offers its importers: the names it binds and the positions it takes."""

import ast
from collections.abc import Iterator, Sequence
from functools import singledispatchmethod


def surface(data: bytes) -> set[str]:
    """The names a module binds at its top level.

    Names bound inside its `if`, `try`, `with`, `for` and `while` blocks count too.
    """
    return set(_bound_in(ast.parse(data).body))


def _bound_in(body: Sequence[ast.stmt]) -> Iterator[str]:
    """The names the statements of `body` bind, those of their blocks included."""
    for node in body:
        yield from _Bindings.of(node)


class _Bindings:
    """The names a top-level statement binds, by the statement's type."""

    @singledispatchmethod
    @staticmethod
    def of(node: ast.stmt) -> Iterator[str]:
        """The names one top-level statement binds; one binding none answers nothing."""
        del node
        yield from ()

    @of.register
    @staticmethod
    def _defined(node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> Iterator[str]:
        yield node.name

    @of.register
    @staticmethod
    def _assigned(node: ast.Assign) -> Iterator[str]:
        for target in node.targets:
            yield from _targets(target)

    @of.register
    @staticmethod
    def _annotated(node: ast.AnnAssign | ast.AugAssign) -> Iterator[str]:
        yield from _targets(node.target)

    @of.register
    @staticmethod
    def _aliased(node: ast.TypeAlias) -> Iterator[str]:
        yield from _targets(node.name)

    @of.register
    @staticmethod
    def _imported(node: ast.Import) -> Iterator[str]:
        yield from (alias.asname or alias.name.split(".")[0] for alias in node.names)

    @of.register
    @staticmethod
    def _imported_from(node: ast.ImportFrom) -> Iterator[str]:
        yield from (alias.asname or alias.name for alias in node.names)

    @of.register
    @staticmethod
    def _looped(node: ast.If | ast.For | ast.AsyncFor | ast.While) -> Iterator[str]:
        yield from _bound_in(node.body)
        yield from _bound_in(node.orelse)

    @of.register
    @staticmethod
    def _withed(node: ast.With | ast.AsyncWith) -> Iterator[str]:
        yield from _bound_in(node.body)

    @of.register
    @staticmethod
    def _tried(node: ast.Try | ast.TryStar) -> Iterator[str]:
        for block in (node.body, node.orelse, node.finalbody):
            yield from _bound_in(block)
        for handler in node.handlers:
            yield from _bound_in(handler.body)


def _targets(target: ast.expr) -> Iterator[str]:
    """The names an assignment target binds, unpacked."""
    match target:
        case ast.Name(id=name):
            yield name
        case ast.Tuple(elts=items) | ast.List(elts=items):
            for item in items:
                yield from _targets(item)
        case ast.Starred(value=inner):
            yield from _targets(inner)


def positional(data: bytes) -> dict[tuple[str, str], int]:
    """How many parameters each top-level function and method takes by position.

    They are keyed by (name, method), with an empty method for a function; methods are those of
    top-level classes.
    """
    counts: dict[tuple[str, str], int] = {}
    for node in ast.parse(data).body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            counts[node.name, ""] = _count(node)
        elif isinstance(node, ast.ClassDef):
            counts |= {
                (node.name, member.name): _count(member)
                for member in node.body
                if isinstance(member, ast.FunctionDef | ast.AsyncFunctionDef)
            }
    return counts


def _count(node: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    return len(node.args.posonlyargs) + len(node.args.args)
