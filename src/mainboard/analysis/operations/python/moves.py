# Moving top-level definitions: into another module, constants into a package's `constants.py`,
# or a constant up to where it can first stand.

import ast
import re
from itertools import groupby
from typing import TYPE_CHECKING, ClassVar, TypeIs

from ...edits import Append, Edit, Import, Insert, Replace
from ...findings import Span
from ..base import Operation
from .imports import Place, Statement
from .source import Block, Module

if TYPE_CHECKING:
    from collections.abc import Collection, Iterable, Mapping, Sequence


class Relocation:
    """One move's two modules: the one the definitions leave and the destination they reach.

    literal: whether only literal assignments move, each one that can moving without the rest.
    """

    @staticmethod
    def _checking(node: ast.stmt) -> TypeIs[ast.If]:
        """Whether `node` is an `if TYPE_CHECKING:` block."""
        return isinstance(node, ast.If) and ast.unparse(node.test) == "TYPE_CHECKING"

    def __init__(self, row: Mapping, *, literal: bool) -> None:
        self.row, self.literal = row, literal
        self.source, self.held = Module(row["content"]), Module(row["destination_content"])
        self.here = Place(row["path"], row["module"])
        self.there = Place(row["destination"], row["destination_module"])

    def _carry(
        self, alias: ast.alias, node: ast.Import | ast.ImportFrom, *, checking: bool
    ) -> str | None:
        """`alias` imported as the destination spells it, empty for one of its own names.

        None for a project name the old module imports for type checking only.
        """
        if isinstance(node, ast.Import):
            return ast.unparse(ast.Import(names=[alias]))
        target = self.here.resolved(node)
        if target == self.there.module:
            return ""
        if checking and (
            node.level or target.partition(".")[0] == self.here.module.partition(".")[0]
        ):
            return None
        spelled = self.there.spelled(target, relative=node.level > 0)
        return f"from {spelled} import {ast.unparse(alias)}"

    def _statements(self) -> list[tuple[ast.stmt, bool]]:
        """The old module's top-level statements, those under `if TYPE_CHECKING:` among them.

        Each comes with whether it is under one.
        """
        return [
            (each, each is not node)
            for node in self.source.tree.body
            for each in (node.body if Relocation._checking(node) else [node])
        ]

    def arrival(self, blocks: Mapping[str, Block], carried: Sequence[str]) -> list[Edit]:
        """The destination's edits: the definitions it does not hold yet, and what they import."""
        fresh = [block.text for block in blocks.values() if not self.held.twin(block)]
        gap, destination = ("\n" if self.literal else "\n\n"), self.row["destination"]
        if not fresh:
            return []
        if self.held.content.strip():
            imports = (Import(path=destination, statement=each) for each in carried)
            return [Append(path=destination, text=gap + gap[1:].join(fresh)), *imports]
        head = "".join(f"{each}\n" for each in sorted(carried)) + (gap if carried else "")
        return [Append(path=destination, text=head + gap[1:].join(fresh))]

    def blocks(self) -> dict[str, Block] | None:
        """The definitions that move, by name, in the order the module defines them.

        A name the destination already uses moves only when it defines it the same way. None when
        a definition cannot move and the move is not of literals alone.
        """
        found = {name: self.source.block(name, literal=self.literal) for name in self.row["names"]}
        movable = {
            name: block
            for name, block in found.items()
            if block is not None and (not self.held.mentions(name) or self.held.twin(block))
        }
        if len(movable) < len(found) and not self.literal:
            return None
        return dict(sorted(movable.items(), key=lambda item: item[1].first))

    def carried(self, names: Collection[str]) -> list[str] | None:
        """The imports the definitions `names` read, spelled from the destination.

        None when one reads a top-level name of the old module that stays, or a project name it
        imports for type checking only.
        """
        reads = {
            each.id
            for node in self.source.tree.body
            if Module.target(node) in names
            for each in ast.walk(node)
            if isinstance(each, ast.Name)
        }
        statements = self._statements()
        importing = [
            (node, checking)
            for node, checking in statements
            if isinstance(node, ast.Import | ast.ImportFrom)
        ]
        if any(
            Module.binds(node) & (reads - set(names))
            for node, _ in statements
            if not isinstance(node, ast.Import | ast.ImportFrom)
        ):
            return None
        carried = [
            self._carry(alias, node, checking=checking)
            for node, checking in importing
            for alias in node.names
            if (alias.asname or alias.name.partition(".")[0]) in reads
        ]
        return None if None in carried else [each for each in carried if each]

    def cycles(self, blocks: Mapping[str, Block]) -> bool:
        """Whether the old module would import back from a destination that imports from it."""
        imported = any(
            isinstance(node, ast.ImportFrom)
            and self.there.resolved(node) == self.here.module
            and any(alias.name not in blocks for alias in node.names)
            for node in ast.walk(self.held.tree)
        )
        return imported and bool(self.source.read(blocks, outside=blocks.values()))

    def departure(self, blocks: Mapping[str, Block]) -> list[Edit]:
        """The old module's edits: each definition gone, and an import of those it still reads."""
        path = self.row["path"]
        read = self.source.read(blocks, outside=blocks.values())
        home = self.here.spelled(self.there.module, relative=True)
        again = (
            [Import(path=path, statement=f"from {home} import {', '.join(read)}")] if read else []
        )
        return [
            *(self.source.cleared(path, first=b.first, last=b.last) for b in blocks.values()),
            *again,
        ]


class MoveDefinitions(Operation):
    """Move top-level definitions out of one module into another, every import of them following.

    Each goes with its decorators and the comments above it to the end of the destination (the
    row's `destination`, dotted `destination_module`), and the imports it reads come along,
    respelled from there. Every `from ... import NAME` of them is respelled, every `module.NAME`
    reads through the destination, and the old module imports back what it still reads. The move
    is whole or refused, with the reason.
    """

    # Whether only literal assignments move, each one that can moving without the rest.
    literal: ClassVar[bool] = False

    @staticmethod
    def _rewritten(
        row: Mapping, moved: Collection[str], *, destination: str
    ) -> list[list[Edit] | None]:
        """Each importing statement's edits, `from ... import` and `module.NAME` alike."""
        imports = groupby(row["importers"] or [], key=lambda one: (one["path"], one["start_line"]))
        reaches = groupby(
            row["reached"] or [], key=lambda one: (one["path"], one["statement_start_line"])
        )
        return [
            *(
                Statement(next(group), destination=destination).respelled(moved)
                for _, group in imports
            ),
            *(
                Statement(group[0], prefix="statement_", destination=destination).rerouted(
                    group, moved=moved
                )
                for group in (list(each) for _, each in reaches)
            ),
        ]

    def edits(self, row: Mapping) -> list[Edit] | None:
        move = Relocation(row, literal=self.literal)
        if not (blocks := move.blocks()):
            self.refused(
                "a definition is no top-level statement of its own, or the destination defines "
                "its name otherwise"
            )
            return None
        if (carried := move.carried(blocks)) is None:
            self.refused(
                "a definition reads a top-level name of its module that stays, or a project "
                "import made for type checking only"
            )
            return None
        if None in (
            changes := MoveDefinitions._rewritten(row, blocks, destination=move.there.module)
        ):
            self.refused(
                "an importer binds the module in a way the move cannot follow (`import a as b`)"
            )
            return None
        if move.cycles(blocks):
            self.refused("the destination imports from the module, which would import it back")
            return None
        rewritten = (edit for edits in changes if edits for edit in edits)
        return [*move.arrival(blocks, carried), *move.departure(blocks), *rewritten]


class MoveConstants(MoveDefinitions):
    """Move the constants other modules use out of one module, into a package's `constants.py`.

    The package is the lowest one holding the module and every module of its own top package that
    uses them (the row's `destination_module`). Only literals move, since nothing else is sure to
    mean the same in another module, and each one that can moves without the rest.
    """

    literal = True


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


class ShareConstants(Operation):
    """Define one private constant several modules repeat once, in a package's `constants.py`.

    Each module's definition goes, and the module imports the public name (`_HOUR` becomes `HOUR`)
    from the destination, which defines it once as its first definition stood, the comments above
    and the imports its annotation reads (`Final`) coming along. Refused when a definition is not
    one literal, two differ once parsed, a module already binds the public name or spells the
    private one where the syntax tree does not see it (a template `.format` fills), or the
    destination defines it otherwise.
    """

    @staticmethod
    def _arrival(row: Mapping, block: Block, *, first: Mapping) -> list[Edit] | None:
        """The destination's edits: `block` under its public name, with what it imports.

        first: the module `block` comes from. None when its annotation reads a project name
        imported for type checking only.
        """
        private = row["name"]
        public = private.lstrip("_")
        arriving = {
            "destination": row["destination"],
            "destination_module": row["destination_module"],
            "destination_content": row["destination_content"],
            "names": [private],
        }
        move = Relocation({**first, **arriving}, literal=True)
        carried = move.carried([private])
        text = re.sub(rf"^{re.escape(private)}\b", public, block.text, count=1, flags=re.MULTILINE)
        renamed = {public: block._replace(text=text)}
        return None if carried is None else move.arrival(renamed, carried)

    @staticmethod
    def _blocks(row: Mapping, definers: Sequence[tuple[Mapping, Module]]) -> list[Block] | None:
        """Each defining module's block of the constant, None when they cannot share it.

        They share it when all define one literal the same way, none binds the public name, and
        the destination defines it that way or not at all.
        """
        private, held = row["name"], Module(row["destination_content"])
        public = private.lstrip("_")
        found = [module.block(private) for _, module in definers]
        blocks = [block for block in found if block is not None]
        values = ShareConstants._values(private, (module for _, module in definers))
        defined = ShareConstants._values(public, [held])
        clash = any(public in module.names() for _, module in definers)
        same = len(values) == 1 and (not defined or defined == values)
        return blocks if len(blocks) == len(found) and same and not clash else None

    @staticmethod
    def _left(
        each: Mapping, module: Module, block: Block, *, destination: str
    ) -> list[Edit] | None:
        """One defining module's edits: its definition gone, its reads of the public name.

        None when the source spells the private name more often than its syntax tree reads it.
        """
        path, private = each["path"], Module.target(ast.parse(block.text).body[0])
        spans = module.spans(private, path=path, outside=block)
        if len(re.findall(rf"\b{re.escape(private)}\b", module.content)) != len(spans) + 1:
            return None
        home = Place(path, each["module"]).spelled(destination, relative=True)
        public = private.lstrip("_")
        imported = [Import(path=path, statement=f"from {home} import {public}")] if spans else []
        return [
            module.cleared(path, first=block.first, last=block.last),
            *(Replace(span=span, text=public) for span in spans),
            *imported,
        ]

    @staticmethod
    def _values(name: str, modules: Iterable[Module]) -> set[str]:
        """The values `modules` assign `name` at their top level, as parsed."""
        return {
            ast.dump(node.value)
            for module in modules
            for node in module.tree.body
            if Module.target(node) == name
            and isinstance(node, ast.Assign | ast.AnnAssign)
            and node.value
        }

    def edits(self, row: Mapping) -> list[Edit] | None:
        definers = [(each, Module(each["content"])) for each in row["definers"]]
        if (blocks := ShareConstants._blocks(row, definers)) is None:
            return None
        destination = row["destination_module"]
        left = [
            ShareConstants._left(each, module, block, destination=destination)
            for (each, module), block in zip(definers, blocks, strict=True)
        ]
        public = row["name"].lstrip("_")
        held = ShareConstants._values(public, [Module(row["destination_content"])])
        arrival = [] if held else ShareConstants._arrival(row, blocks[0], first=definers[0][0])
        if arrival is None or None in left:
            return None
        return [*arrival, *(edit for edits in left if edits for edit in edits)]
