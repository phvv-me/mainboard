# One move's two modules: the definitions leave one and reach the other, and what each edits.

import ast
import re
from itertools import groupby, takewhile
from typing import TYPE_CHECKING, TypeIs

from ....claims import Claim
from ....edits import Append, Delete, Edit, Import, Replace
from ....findings import Span
from ..imports import Place
from ..source import Block, Module

if TYPE_CHECKING:
    from collections.abc import Collection, Iterable, Mapping, Sequence


class Relocation:
    """One move's two modules: the one the definitions leave and the destination they reach.

    literal: whether only assignments of a literal move.
    compact: whether the definitions are constants, packed one per line and each one that can
        moving without the rest; unset, whatever `literal` is.
    """

    @staticmethod
    def _checking(node: ast.stmt) -> TypeIs[ast.If]:
        """Whether `node` is an `if TYPE_CHECKING:` block."""
        return isinstance(node, ast.If) and ast.unparse(node.test) == "TYPE_CHECKING"

    @staticmethod
    def _named(paragraph: Block, among: Iterable[str]) -> set[str]:
        """The names of `among` that `paragraph` spells in backticks, `Mode` or `Mode.FAST`."""
        return {name for name in among if re.search(rf"`{re.escape(name)}\b", paragraph.text)}

    def __init__(self, row: Mapping, *, literal: bool, compact: bool | None = None) -> None:
        self.row, self.literal = row, literal
        self.compact = literal if compact is None else compact
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

    def _header(self) -> tuple[int, list[Block]]:
        """The comment that opens the old module: its length in lines and its paragraphs.

        A bare `#` parts the paragraphs. A comment with a statement right below it is that
        statement's own, and no header.
        """
        lines = self.source.content.splitlines(keepends=True)
        opening = list(takewhile(lambda line: line.startswith("#"), lines))
        if len(opening) < len(lines) and lines[len(opening)].strip():
            return 0, []
        parted = groupby(enumerate(opening, 1), key=lambda item: item[1].strip() == "#")
        groups = [list(group) for bare, group in parted if not bare]
        text = ["".join(line for _, line in group) for group in groups]
        return len(opening), [
            Block(group[0][0], group[-1][0], each, "")
            for group, each in zip(groups, text, strict=True)
        ]

    def _statements(self) -> list[tuple[ast.stmt, bool]]:
        """The old module's top-level statements, those under `if TYPE_CHECKING:` among them.

        Each comes with whether it is under one.
        """
        return [
            (each, each is not node)
            for node in self.source.tree.body
            for each in (node.body if Relocation._checking(node) else [node])
        ]

    def _unheaded(self, blocks: Iterable[Block]) -> list[Edit]:
        """The edit taking from the old module's header the paragraphs that go with `blocks`."""
        length, paragraphs = self._header()
        gone = {head.first for block in blocks for head in block.heads}
        if not gone:
            return []
        lines, path = self.source.content.splitlines(), self.row["path"]
        kept = "#\n".join(each.text for each in paragraphs if each.first not in gone)
        if kept:
            end = len(lines[length - 1].encode())
            span = Span(path=path, start_line=1, end_line=length, end_column=end)
            return [Replace(span=span, text=kept.rstrip("\n"))]
        blank = len(list(takewhile(lambda line: not line.strip(), lines[length:])))
        return [Delete(span=Span(path=path, start_line=1, end_line=length + blank))]

    def arrival(self, blocks: Mapping[str, Block], carried: Sequence[str]) -> list[Edit]:
        """The destination's edits: the definitions it does not hold yet, and what they import.

        A destination with no text yet is claimed, so that a second move into it waits for the
        first one's imports to stand there instead of writing its own into the middle.
        """
        fresh = [block.whole for block in blocks.values() if not self.held.twin(block)]
        gap, destination = ("\n" if self.compact else "\n\n"), self.row["destination"]
        if not fresh:
            return []
        if self.held.content.strip():
            imports = (Import(path=destination, statement=each) for each in carried)
            return [Append(path=destination, text=gap + gap[1:].join(fresh)), *imports]
        head = "".join(f"{each}\n" for each in sorted(carried)) + (gap if carried else "")
        return [
            Claim(span=Span(path=destination, start_line=1)),
            Append(path=destination, text=head + gap[1:].join(fresh)),
        ]

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
        if len(movable) < len(found) and not self.compact:
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
            *self._unheaded(blocks.values()),
            *again,
        ]

    def headed(self, blocks: Mapping[str, Block]) -> dict[str, Block] | None:
        """`blocks` with the paragraphs of the old module's header that name them.

        A paragraph names a definition in backticks and goes with the first one it names. None
        when one names a definition that moves and one that stays, which no move can part.
        """
        paragraphs = self._header()[1]
        staying = {Module.target(node) for node in self.source.tree.body} - {*blocks, ""}
        moving = [Relocation._named(each, blocks) for each in paragraphs]
        if any(
            m and Relocation._named(p, staying) for p, m in zip(paragraphs, moving, strict=True)
        ):
            return None
        order = list(blocks)
        owners = [min(named, key=order.index, default="") for named in moving]
        return {
            name: block._replace(
                heads=tuple(
                    p for p, owner in zip(paragraphs, owners, strict=True) if owner == name
                )
            )
            for name, block in blocks.items()
        }
