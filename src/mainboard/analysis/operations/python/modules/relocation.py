# One module's imports after a move: its from-imports respelled, and a plain `import a.b` of a
# moved module with the `a.b.NAME` reads through it following the module to its new name.

import ast
from typing import TYPE_CHECKING

from ....edits import Edit, Replace
from ....findings import Span
from ..imports import Place

if TYPE_CHECKING:
    from collections.abc import Collection, Iterable, Mapping


class ModuleRelocation:
    """One module's imports respelled after `moves` (old dotted module to new).

    The module's own relative imports are read from where it stood and written from where it
    lands. A plain `import a.b` of a moved module follows it, as does every `a.b.NAME` the module
    reads through a plain import of `a`. A move that would bind another top package than the one
    such an import binds (`a` becoming `c`) is refused, since the module's other reads of `a` would
    lose their meaning.
    """

    @classmethod
    def respelled(cls, files: Iterable[Mapping], moves: Mapping[str, str]) -> list[Edit] | None:
        """The edits respelling the imports of every module in `files`, None when one refuses.

        Each file is a record of its `path`, `module` and `content`. None too for a file that does
        not parse.
        """
        made: list[Edit] = []
        for file in files:
            try:
                relocation = cls(moves, Place(file["path"], file["module"]), file["content"])
            except SyntaxError:
                return None
            if (rewritten := relocation.edits()) is None:
                return None
            made += rewritten
        return made

    @staticmethod
    def _links(top: ast.Attribute) -> tuple[ast.expr, list[ast.Attribute]]:
        """The root of the chain of attributes ending in `top`, and its attributes root first."""
        attributes = [top]
        while isinstance(inner := attributes[-1].value, ast.Attribute):
            attributes.append(inner)
        return attributes[-1].value, attributes[::-1]

    def __init__(self, moves: Mapping[str, str], here: Place, content: str) -> None:
        """here: where the module stands before the moves; content: its source."""
        self.moves, self.here = moves, here
        self.there = Place(here.path, moves.get(here.module, here.module))
        self.tree = ast.parse(content)
        self.lines = content.encode().split(b"\n")

    def _chain(self, top: ast.Attribute, plain: Collection[str]) -> list[Edit] | None:
        """The edit moving the module the chain ending in `top` reads, none when it reads none.

        None when the new name would bind another top package than the root name does.
        """
        root, attributes = ModuleRelocation._links(top)
        if not isinstance(root, ast.Name) or root.id not in plain:
            return []
        names = [root.id, *(each.attr for each in attributes)]
        dotted = (".".join(names[:depth]) for depth in range(len(names), 1, -1))
        old = next((module for module in dotted if module in self.moves), None)
        if old is None:
            return []
        last = attributes[old.count(".") - 1]
        if self.moves[old].partition(".")[0] != root.id or last.end_lineno != root.lineno:
            return None
        return [Replace(span=self._span(root, last=last), text=self.moves[old])]

    def _dotted(self, plain: Collection[str]) -> list[Edit] | None:
        """The edits respelling the moved modules the file reads as `a.b.NAME` through `import a`.

        `plain` holds the names its plain imports bind. A read is the longest chain of attributes,
        rooted in such a name, that starts with a moved module.
        """
        attributes = [node for node in ast.walk(self.tree) if isinstance(node, ast.Attribute)]
        inner = {id(node.value) for node in attributes}
        made: list[Edit] = []
        for top in (node for node in attributes if id(node) not in inner):
            if (edits := self._chain(top, plain)) is None:
                return None
            made += edits
        return made

    def _from(self, node: ast.ImportFrom) -> list[Edit]:
        """The edit respelling a `from ... import` after the moves, none when they leave it be.

        A moved module imported by name keeps the name it was bound to (`base as cuda_base`), and
        an explicit alias stays, since `NAME as NAME` is a re-export.
        """
        base = self.here.resolved(node)
        groups: dict[str, list[str]] = {}
        for alias in node.names:
            bound = alias.asname or alias.name
            if (whole := f"{base}.{alias.name}") in self.moves:
                target, name = self.moves[whole].rsplit(".", 1)
            else:
                target, name = self.moves.get(base, base), alias.name
            written = name if alias.asname is None and name == bound else f"{name} as {bound}"
            groups.setdefault(target, []).append(written)
        spelled = {
            self.there.spelled(target, relative=node.level > 0): names
            for target, names in groups.items()
        }
        kept = [
            alias.name + (f" as {alias.asname}" if alias.asname else "") for alias in node.names
        ]
        if spelled == {"." * node.level + (node.module or ""): kept}:
            return []
        text = ("\n" + " " * node.col_offset).join(
            f"from {spelling} import {', '.join(names)}" for spelling, names in spelled.items()
        )
        return [Replace(span=self._span(node, last=node), text=text)]

    def _named(self, alias: ast.alias) -> Span | None:
        """Where `alias` spells its dotted name, None when its columns do not hold that name."""
        name, start = alias.name.encode(), alias.col_offset
        if self.lines[alias.lineno - 1][start : start + len(name)] != name:
            return None
        return Span(
            path=self.here.path,
            start_line=alias.lineno,
            start_column=start,
            end_line=alias.lineno,
            end_column=start + len(name),
        )

    def _plain(self, node: ast.Import) -> list[Edit] | None:
        """The edits naming each moved module `node` imports by its new name.

        None when an import that binds the top package would bind another.
        """
        made: list[Edit] = []
        for alias in (each for each in node.names if each.name in self.moves):
            new = self.moves[alias.name]
            rebound = not alias.asname and new.partition(".")[0] != alias.name.partition(".")[0]
            if rebound or (span := self._named(alias)) is None:
                return None
            made.append(Replace(span=span, text=new))
        return made

    def _span(self, first: ast.expr | ast.stmt, *, last: ast.expr | ast.stmt) -> Span:
        """Where the source from the start of `first` to the end of `last` stands."""
        return Span(
            path=self.here.path,
            start_line=first.lineno,
            start_column=first.col_offset,
            end_line=last.end_lineno,
            end_column=last.end_col_offset,
        )

    def edits(self) -> list[Edit] | None:
        """The edits respelling each import of a moved module and each dotted read through it.

        That includes any statement whose own spelling moved with the module. None when one cannot
        be made.
        """
        statements = [
            node for node in ast.walk(self.tree) if isinstance(node, ast.Import | ast.ImportFrom)
        ]
        plain = {
            alias.name.partition(".")[0]
            for node in statements
            if isinstance(node, ast.Import)
            for alias in node.names
            if not alias.asname
        }
        parts = [
            self._plain(node) if isinstance(node, ast.Import) else self._from(node)
            for node in statements
        ]
        made: list[Edit] = []
        for part in [*parts, self._dotted(plain)]:
            if part is None:
                return None
            made += part
        return made
