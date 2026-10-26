"""The import edits a rule's file moves make, shared by the rules that move or rename modules."""

import ast

from .... import edits
from ....edits import Edit
from ....findings import Hit, Span


class ModuleRelocation:
    """The import statements of one module, respelled after `moves` (old module to new)."""

    def __init__(self, moves: dict[str, str], module: str, *, package: bool) -> None:
        self.moves, self.package = moves, package
        self.old = module
        self.new = moves.get(module, module)

    def _base(self, node: ast.ImportFrom) -> str:
        """The absolute module `node` imports from, read from where this module stood."""
        if not node.level:
            return node.module or ""
        parts = self.old.split(".") if self.package else self.old.split(".")[:-1]
        parts = parts[: len(parts) - (node.level - 1)]
        return ".".join([*parts, *([node.module] if node.module else [])])

    def _respelled(self, node: ast.ImportFrom) -> str | None:
        """Respell the statement after the moves, or return None when they leave it as it is.

        A moved module imported by name keeps the name it was bound to (`base as cuda_base`), and
        an explicit alias stays, since `NAME as NAME` is a re-export.
        """
        base = self._base(node)
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
            (self._spelling(target) if node.level else target): names
            for target, names in groups.items()
        }
        kept = [
            alias.name + (f" as {alias.asname}" if alias.asname else "") for alias in node.names
        ]
        if spelled == {"." * node.level + (node.module or ""): kept}:
            return None
        return ("\n" + " " * node.col_offset).join(
            f"from {spelling} import {', '.join(names)}" for spelling, names in spelled.items()
        )

    def _spelling(self, target: str) -> str:
        """`target` relative to where this module now stands."""
        base = self.new.split(".") if self.package else self.new.split(".")[:-1]
        goal = target.split(".")
        common = 0
        while common < min(len(base), len(goal)) and base[common] == goal[common]:
            common += 1
        return "." * (len(base) - common + 1) + ".".join(goal[common:])

    def edits(self, path: str, *, content: str) -> list[Edit] | None:
        """Return one edit per statement that names a moved module or whose own spelling moved.

        None when a plain `import` names a moved module, since its dotted uses would move too.
        """
        made: list[Edit] = []
        for node in ast.walk(ast.parse(content)):
            match node:
                case ast.Import(names=names) if any(alias.name in self.moves for alias in names):
                    return None
                case ast.ImportFrom():
                    text = self._respelled(node)
                    if text is not None:
                        span = Span(
                            path=path,
                            start_line=node.lineno,
                            start_column=node.col_offset,
                            end_line=node.end_lineno,
                            end_column=node.end_col_offset,
                        )
                        made.append(edits.replace(span, text))
        return made


def respelled(hit: Hit, moves: dict[str, str]) -> list[Edit] | None:
    """The edits respelling the imports of every module the moves touch, from the hit's `files`.

    Each file of `files` is a record of its `path`, `module`, `package` flag and `content`. None
    when a plain `import` names a moved module, or a file does not parse.
    """
    made: list[Edit] = []
    for file in hit.records("files"):
        relocation = ModuleRelocation(moves, str(file["module"]), package=bool(file["package"]))
        try:
            rewritten = relocation.edits(str(file["path"]), content=str(file["content"]))
        except SyntaxError:
            return None
        if rewritten is None:
            return None
        made += rewritten
    return made
