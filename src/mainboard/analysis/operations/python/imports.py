# One import statement a fix row quotes, and what becomes of it once constants leave a module or a
# name it imports is renamed.

import ast
import re
from typing import TYPE_CHECKING, NamedTuple

from ...constants import INITIALIZER_PATH
from ...edits import Edit, Import, Replace
from ...findings import Span
from .source import Module

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping, Sequence


class Place(NamedTuple):
    """A module by its file and dotted name, and how it spells an import of another module."""

    path: str
    module: str

    @property
    def package(self) -> list[str]:
        """The package a relative import here starts from: the module itself for an initializer."""
        here = self.module.split(".")
        return here if re.search(INITIALIZER_PATH, self.path) else here[:-1]

    def resolved(self, node: ast.ImportFrom) -> str:
        """The dotted module `node` imports from, its leading dots read from here."""
        if not node.level:
            return node.module or ""
        base = self.package[: len(self.package) - node.level + 1]
        return ".".join([*base, *([node.module] if node.module else [])])

    def spelled(self, target: str, *, relative: bool) -> str:
        """The dotted module `target` as an import here spells it, from here when `relative`.

        A module in no package spells every import in full.
        """
        here = self.package
        if not relative or not here:
            return target
        there = target.split(".")
        common = next(
            (i for i, (a, b) in enumerate(zip(here, there, strict=False)) if a != b),
            min(len(here), len(there)),
        )
        return "." * (len(here) - common + 1) + ".".join(there[common:])


class Statement:
    """One import statement a fix row quotes, and its edits once names move to `destination`.

    destination: the dotted module the moved names now live in, or the module itself when the
    names are only renamed.
    """

    @staticmethod
    def spanned(record: Mapping, *, prefix: str = "") -> Span:
        """The span `record` names under columns that start with `prefix`."""
        return Span(
            path=record["path"],
            start_line=record[f"{prefix}start_line"],
            start_column=record[f"{prefix}start_column"],
            end_line=record[f"{prefix}end_line"],
            end_column=record[f"{prefix}end_column"],
        )

    def __init__(self, record: Mapping, *, prefix: str = "", destination: str) -> None:
        """Read the statement a fix row quotes.

        record: the row quoting the statement's lines, its span under columns `prefix`ed, and the
        importing module (`importer`).
        """
        lines = record["statement"].split("\n")
        lines[-1] = lines[-1][: record[f"{prefix}end_column"]]
        lines[0] = lines[0][record[f"{prefix}start_column"] :]
        parsed = ast.parse("\n".join(lines)).body[0]
        self.node = parsed if isinstance(parsed, ast.Import | ast.ImportFrom) else None
        self.destination = destination
        self.place = Place(record["path"], record["importer"])
        self.span = Statement.spanned(record, prefix=prefix)

    def _binding(self, bound: str) -> ast.alias | None:
        """The name this statement binds as `bound`, None when it is no binding to follow."""
        names = self.node.names if self.node is not None else []
        alias = next((each for each in names if (each.asname or each.name) == bound), None)
        aliased = isinstance(self.node, ast.Import) and alias is not None and alias.asname
        return None if aliased else alias

    def _kept(self, alias: ast.alias, statement: str, *, others: int, held: bool) -> list[Edit]:
        """What becomes of this statement once `alias`'s moved names read the constants module.

        others: the reads of the old module through `alias` that stay.
        held: whether the importer already imports the constants module as `statement` does.
        """
        added: list[Edit] = [] if held else [Import(path=self.span.path, statement=statement)]
        if others > 0 or self.node is None:
            return added
        if len(self.node.names) > 1:
            return [Replace(span=self.span, text=self._without(self.node, [alias])), *added]
        return [Replace(span=self.span, text="" if held else statement)]

    def _named(self, alias: ast.alias) -> Span:
        """Where `alias` spells the name it imports, which no `as` precedes."""
        first = alias.lineno == 1
        line = self.span.start_line + alias.lineno - 1
        column = alias.col_offset + (self.span.start_column if first else 0)
        return Span(
            path=self.span.path,
            start_line=line,
            start_column=column,
            end_line=line,
            end_column=column + len(alias.name.encode()),
        )

    def _reaching(self) -> tuple[str, str]:
        """The import that reaches the destination where this statement reached the old module.

        Beside it comes the prefix a moved name is then read through.
        """
        if isinstance(self.node, ast.ImportFrom):
            package, _, leaf = self.destination.rpartition(".")
            spelled = self.place.spelled(package, relative=self.node.level > 0)
            return f"from {spelled} import {leaf}", leaf
        return f"import {self.destination}", self.destination

    def _without(self, node: ast.Import | ast.ImportFrom, dropped: Collection[ast.alias]) -> str:
        """The statement as written without the names `dropped`."""
        node.names = [alias for alias in node.names if alias not in dropped]
        return ast.unparse(node)

    def binds(self, name: str) -> bool:
        """Whether this `from ... import` binds `name` itself, no `as` giving it another name."""
        return isinstance(self.node, ast.ImportFrom) and any(
            alias.name == name and alias.asname is None for alias in self.node.names
        )

    def renamed(self, old: str, *, new: str) -> list[Replace] | None:
        """The edits importing `old` as `new` where this `from ... import` takes it.

        Each replaces the imported name alone, so an `as` alias keeps the name bound. None for a
        statement that does not import `old`.
        """
        names = self.node.names if isinstance(self.node, ast.ImportFrom) else []
        taken = [alias for alias in names if alias.name == old]
        return [Replace(span=self._named(alias), text=new) for alias in taken] or None

    def rerouted(self, reaches: Sequence[Mapping], *, moved: Collection[str]) -> list[Edit] | None:
        """The moved `module.NAME` uses of `reaches` read through the constants module.

        The statement is respelled when nothing else of the old module is read through it, and
        the constants module is imported beside it otherwise. None for a binding it cannot follow.
        """
        bound, importer = reaches[0]["binding"], Module(reaches[0]["content"])
        if (alias := self._binding(bound)) is None:
            return None
        statement, prefix = self._reaching()
        held = statement in importer.content
        if "." not in prefix and not held and prefix in importer.names():
            return None
        respelled: list[Edit] = [
            Replace(span=Statement.spanned(reach), text=f"{prefix}.{reach['name']}")
            for reach in reaches
            if reach["name"] in moved
        ]
        if not respelled:
            return []
        others = importer.uses(bound) - len(respelled)
        return [*respelled, *self._kept(alias, statement, others=others, held=held)]

    def respelled(self, moved: Collection[str]) -> list[Edit] | None:
        """This `from ... import`'s moved names imported from the constants module instead.

        None for a statement that is not a plain `from ... import`.
        """
        if not isinstance(self.node, ast.ImportFrom):
            return None
        taken = [alias for alias in self.node.names if alias.name in moved]
        module = self.place.spelled(self.destination, relative=self.node.level > 0)
        home = self.place.module == self.destination
        statement = "" if home else f"from {module} import {', '.join(map(ast.unparse, taken))}"
        if not taken or len(taken) == len(self.node.names):
            return [Replace(span=self.span, text=statement)] if taken else []
        if home:
            return [Replace(span=self.span, text=self._without(self.node, taken))]
        rest = self._without(self.node, taken)
        return [
            Replace(span=self.span, text=rest),
            Import(path=self.span.path, statement=statement),
        ]
