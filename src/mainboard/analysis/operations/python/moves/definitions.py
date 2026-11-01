# Moving top-level definitions out of one module into another, constants into a package's
# `constants.py`.

from itertools import groupby
from typing import TYPE_CHECKING, ClassVar

from ....edits import Edit
from ...base import Operation
from ..imports import Statement
from .relocation import Relocation

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping

    from ..source import Block


class MoveDefinitions(Operation):
    """Move top-level definitions out of one module into another, every import of them following.

    Each goes with its decorators, the comments above it and those ending its body to the end of
    the destination (the row's `destination`, dotted `destination_module`), and the imports it
    reads come along, respelled from there. A paragraph of the old module's header comment that
    names a moved definition in backticks goes with it, above it. Every `from ... import NAME` of
    them is respelled, every `module.NAME` reads through the destination, and the old module
    imports back what it still reads. Moves into one destination that has no text yet take it in
    turns, one per pass. The move is whole or refused, with the reason.
    """

    # Whether only assignments of a literal move.
    literal: ClassVar[bool] = False
    # Whether the definitions are constants, packed one per line and each one that can moving
    # without the rest.
    compact: ClassVar[bool] = False

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

    def _blocks(self, move: Relocation) -> dict[str, Block] | None:
        """The definitions that move with their header paragraphs, None with the reason."""
        if not move.there.module:
            self.refused(
                "the destination stands outside the source root, where no module name reaches it"
            )
            return None
        if not (found := move.blocks()):
            self.refused(
                "a definition is no top-level statement of its own, or the destination defines "
                "its name otherwise"
            )
            return None
        if (blocks := move.headed(found)) is None:
            self.refused(
                "a header comment names a definition that moves and one that stays, which no "
                "move can part"
            )
        return blocks

    def edits(self, row: Mapping) -> list[Edit] | None:
        move = Relocation(row, literal=self.literal, compact=self.compact)
        if (blocks := self._blocks(move)) is None:
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
    uses them (the row's `destination_module`). A constant of any value moves with the imports its
    value reads, and each one that can moves without the rest.
    """

    compact = True
