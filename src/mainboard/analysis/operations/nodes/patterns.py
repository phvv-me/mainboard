# Declared rewrites of a node: a pattern and a template in the language's own syntax, what each
# hole of the pattern took, and the template that renders those captures where the node stood.

import json
import re
from collections.abc import Mapping
from typing import TYPE_CHECKING, Self

from patos import FrozenModel
from pydantic import model_validator

from .dialects import Dialect, Python

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .site import Node, Tree

# A hole: `$$$X` a run of sibling nodes, `$X` one node, `$_` any node that is not kept.
_HOLES = re.compile(r"\$\$\$(\w+)|\$(\w+)")
# One `parse_ast` row as the struct `Node` validates. The parse prunes nothing and numbers its
# semantic types, and its columns are one-based where the facts' are not.
_PARSED = (
    "{'node': n.node_id, 'parent': n.parent_id, 'index': n.sibling_index, 'type': n.type, "
    "'semantic': semantic_type_to_string(n.semantic_type), 'start': n.start_byte, "
    "'end': n.end_byte, 'line': n.start_line, 'column': n.start_column - 1, "
    "'end_line': n.end_line, 'end_column': n.end_column - 1}"
)


def _names(text: str) -> set[str]:
    """The names of the holes `text` holds."""
    return {match[1] or match[2] for match in _HOLES.finditer(text)}


class Capture(FrozenModel):
    """What a hole took: the bytes `[start, end)` of the target, and the ids of the nodes in them.

    A run's bytes reach over the comments it owns. A hole that took nothing has no nodes.
    """

    start: int
    end: int
    nodes: tuple[int, ...]


class Rewrite(FrozenModel):
    """A declared rewrite: the first pattern that unifies with a node is rendered through `to`.

    language: the dialect of the files it rewrites, by the facts' name for it.
    match: the pattern, a statement or an expression of that language with holes. A name held
        twice must capture equal text.
    to: the template put where the node stood, naming the holes it keeps. A hole it leaves out is
        dropped with its text.
    imports: the statements `to` needs (Python only).
    where: the texts a hole's capture must be one of, else the pattern does not unify
        (`{"INERT": ("pass", "...")}`).
    """

    language: str = "python"
    match: str
    to: str
    imports: tuple[str, ...] = ()
    where: Mapping[str, tuple[str, ...]] = {}

    @classmethod
    def bindings(cls, rewrites: Sequence[Rewrite]) -> dict[str, str]:
        """The parameters `fix(rewrites)` reads: the rewrites as JSON, each pattern and grammar."""
        return {"rewrites": json.dumps([rewrite.model_dump() for rewrite in rewrites])} | {
            name: text
            for number, rewrite in enumerate(rewrites)
            for name, text in (
                (f"pattern_{number}", rewrite.expando()),
                (f"grammar_{number}", rewrite.grammar()),
            )
        }

    @classmethod
    def fix(cls, rewrites: Sequence[Rewrite]) -> str:
        """The fix SQL of a rule declaring `rewrites`, over the parameters `bindings` gives it.

        One row per finding that names a `node`, holding the rewrites as JSON and the parse of each
        pattern.
        """
        parsed = ", ".join(
            f"(SELECT list({_PARSED} ORDER BY n.node_id) FROM "
            f"parse_ast($pattern_{number}, $grammar_{number}, source := 'full') n)"
            for number in range(len(rewrites))
        )
        return (
            "SELECT finding_key, 'rewrite-node' AS operation, path, node, "
            f"$rewrites AS rewrites, [{parsed}] AS patterns FROM finding WHERE node IS NOT NULL"
        )

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        """Refuse lower case or unbound hole names, and imports outside Python."""
        names = _names(self.match)
        if lower := {name for name in names if name != name.upper()}:
            raise ValueError(f"hole names are upper case, not {sorted(lower)}")
        if unbound := (_names(self.to) | self.where.keys()) - (names - {"_"}):
            raise ValueError(f"{sorted(unbound)} are no holes of the pattern {self.match!r}")
        if self.imports and self.language != "python":
            raise ValueError(f"imports are Python statements, not {self.language}")
        return self

    def admits(self, texts: Mapping[str, str]) -> bool:
        """Whether the captured `texts` are those `where` allows."""
        return all(texts.get(name) in allowed for name, allowed in self.where.items())

    def expando(self) -> str:
        """The pattern with each `$` the letter its language reads as an identifier."""
        return self.match.replace("$", Dialect.find(self.language, attr="language").expando)

    def grammar(self) -> str:
        """The tree-sitter grammar its language parses with."""
        return Dialect.find(self.language, attr="language").grammar


class Template:
    """A template's text with holes, rendered where a node stood.

    The first line goes where the node starts and the others at the node's own indent, each
    capture moved to the column its hole stands at. A lone expression hole is parenthesized where
    its parent binds tighter and the expression is no atom, where its parent's braces would read
    its leading `{` as an escape, and in Python where it breaks lines that only the brackets it
    left covered.
    """

    @staticmethod
    def _cuts(line: str, shift: int) -> bool:
        """Whether moving `line` left by `-shift` columns would cut text."""
        return shift < 0 and bool(line[:-shift].strip())

    @staticmethod
    def _dropped(line: str, captures: Mapping[str, Capture]) -> bool:
        """Whether `line` holds only a `$$$` hole that took nothing."""
        lone = _HOLES.fullmatch(line.strip())
        return lone is not None and bool(lone[1]) and not captures[lone[1]].nodes

    @staticmethod
    def _open(capture: Capture, tree: Tree) -> bool:
        """Whether the capture ends in a comment, which would swallow the text after it."""
        inside = tree.within(capture.start, end=capture.end)
        return any(node.comment and node.end == capture.end for node in inside)

    @staticmethod
    def _shifted(line: str, shift: int) -> str:
        """`line` moved `shift` columns, a blank one left as it is."""
        if not line.strip():
            return line
        return " " * shift + line if shift >= 0 else line[-shift:]

    def __init__(self, text: str, *, dialect: Dialect) -> None:
        self.text, self.dialect = text, dialect

    def _line(
        self, number: int, line: str, captures: Mapping[str, Capture], tree: Tree, base: int
    ) -> str | None:
        """Template line `number` with its holes filled.

        None when a capture cannot move, or ends in a comment with template text after it.
        """
        lead = len(line) - len(line.lstrip(" "))
        pieces, kept = [], 0
        for match in _HOLES.finditer(line):
            capture = captures[match[1] or match[2]]
            shift = base + lead - (tree.indent(capture.start) if capture.nodes else 0)
            moved = self._moved(capture, tree, shift)
            if moved is None or (line[match.end() :].strip() and self._open(capture, tree)):
                return None
            pieces += [line[kept : match.start()], moved]
            kept = match.end()
        joined = "".join([*pieces, line[kept:]])
        return joined if number == 0 or not joined.strip() else " " * base + joined

    def _moved(self, capture: Capture, tree: Tree, shift: int) -> str | None:
        """The capture's text with its later lines moved `shift` columns.

        None when that would change the text: a line would lose non-blank characters, or a string
        holding line breaks would be re-indented.
        """
        text = tree.data[capture.start : capture.end].decode()
        if not shift or "\n" not in text:
            return text
        strings = self.dialect.strings
        inside = tree.within(capture.start, end=capture.end)
        first, *rest = text.split("\n")
        if any(self._cuts(line, shift) for line in rest) or any(
            node.type in strings and node.line != node.end_line for node in inside
        ):
            return None
        return "\n".join([first, *(self._shifted(line, shift) for line in rest)])

    def _torn(self, capture: Capture, tree: Tree, parent: Node) -> bool:
        """Whether an expression in a Python statement breaks a line that no bracket covers.

        A line break ends a statement there unless a bracket or a string of the capture itself
        covers it, so the brackets of the call the capture left would have to stay.
        """
        statement = parent.type in self.dialect.blocks or parent.parent is None
        if not isinstance(self.dialect, Python) or statement:
            return False
        inside = tree.within(capture.start, end=capture.end)
        covered = [(node.start, node.end) for node in inside if node.type in self.dialect.strings]
        opened: list[int] = []
        for node in (node for node in inside if node.semantic == "PARSER_DELIMITER"):
            if tree.text(node) in (b"(", b"[", b"{"):
                opened.append(node.end)
            elif opened:
                covered.append((opened.pop(), node.start))
        text = tree.data[capture.start : capture.end]
        breaks = [capture.start + place for place, byte in enumerate(text) if byte == 10]
        return any(not any(start <= place < end for start, end in covered) for place in breaks)

    def _wrapped(
        self, text: str, captures: Mapping[str, Capture], tree: Tree, parent: Node | None
    ) -> str:
        """`text` in parentheses when it is one bare expression hole that needs them.

        That is when its parent binds tighter and the expression is no atom, when it breaks lines
        that only the brackets it left covered, or when it opens with a `{` that its parent's
        braces would read as an escaped one (`f"{{x: 1}}"` is text, not a dictionary).
        """
        lone = _HOLES.fullmatch(self.text.strip())
        if lone is None or lone[1] or parent is None:
            return text
        capture = captures[lone[2]]
        taken = [node for node in tree.nodes if node.node in capture.nodes]
        if len(taken) != 1:
            return text
        tight = parent.type in self.dialect.binding and taken[0].type not in self.dialect.atoms
        escaped = parent.type in self.dialect.escaping and text.startswith("{")
        return f"({text})" if tight or escaped or self._torn(capture, tree, parent) else text

    def holes(self) -> set[str]:
        """The names of the holes the template keeps."""
        return _names(self.text)

    def rendered(
        self, captures: Mapping[str, Capture], tree: Tree, at: Node, *, parent: Node | None
    ) -> str | None:
        """The text that replaces `at`, None when a capture cannot move to where its hole stands.

        parent: the parent of `at`, which decides whether a lone expression hole needs parentheses.
        """
        base = tree.indent(at.start)
        lines = []
        for number, line in enumerate(self.text.split("\n")):
            if self._dropped(line, captures):
                continue
            if (filled := self._line(number, line, captures, tree, base)) is None:
                return None
            lines.append(filled)
        return self._wrapped("\n".join(lines), captures, tree, parent)
