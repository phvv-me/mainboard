# Defining a private constant several modules repeat once, in a package's `constants.py`.

import ast
import re
from typing import TYPE_CHECKING

from ....edits import Edit, Import, Replace
from ...base import Operation
from ..imports import Place
from ..source import Block, Module
from .relocation import Relocation

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence


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
