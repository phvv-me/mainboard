# The tokens of a Python source that spell a name, and which of them a rename leaves unsaid.

import ast
import io
import re
import tokenize
from typing import TYPE_CHECKING, NamedTuple

from ....findings import Span

if TYPE_CHECKING:
    from collections.abc import Collection

# A place in the source: its line and its column in bytes.
type Position = tuple[int, int]

# The brackets of an expression, and what follows the `=` that makes a field print its source.
_CLOSING, _OPENING = frozenset(")]}"), frozenset("([{")
_AFTER_SPECIFIER = frozenset({"}", "!", ":"})


class Token(NamedTuple):
    """Where one token spelling a name starts and ends, and whether it is an identifier."""

    start: Position
    end: Position
    name: bool


class Spelling:
    """The tokens of a Python source that spell a name, and what its syntax tree makes of each.

    A rename says a token when it rewrites it. A token it leaves must be a name the tree places
    apart from the renamed one: an attribute, a keyword, a variable of another scope. Any other
    (a string, a `global` statement, `except ... as`, an import alias) is a spelling the rename
    does not follow, and so is a name in an f-string field printing its own source (`{name=}`),
    which a rename would rewrite in the string's value.
    """

    @staticmethod
    def _accounted(
        token: Token, starts: Collection[Position], *, placed: Collection[Position]
    ) -> bool:
        """Whether a rename leaves nothing of `token` unsaid.

        An identifier is rewritten or placed by the tree. A string is rewritten when a span
        starts inside it.
        """
        if token.name:
            return token.start in placed or token.start in starts
        return any(token.start <= start < token.end for start in starts)

    @staticmethod
    def _spells(literal: str, *, old: str) -> bool:
        """Whether the string token `literal` is `old`, or a dotted or colon path ending in it."""
        value = ast.literal_eval(literal)
        return (
            isinstance(value, str)
            and re.fullmatch(rf"(?:[^\n]*[.:])?{re.escape(old)}", value) is not None
        )

    @staticmethod
    def tail(node: ast.Attribute) -> Position:
        """Where the attribute name that ends `node` starts."""
        return node.end_lineno or node.lineno, (node.end_col_offset or 0) - len(node.attr.encode())

    def __init__(self, content: str, tree: ast.Module) -> None:
        self.content, self.tree = content, tree
        self.lines = content.split("\n")
        self.lexed = list(tokenize.generate_tokens(io.StringIO(content).readline))

    def _documented(self, old: str) -> set[Position]:
        """Where `old` is spelled in an f-string field that prints its own source (`{old + 1=}`).

        The text of such a field is part of the string, so renaming anything in it changes the
        string's value. Only that field has an `=` directly before its `}`, `!` or `:`.
        """
        return {
            place
            for at, token in enumerate(self.lexed)
            if token.type == tokenize.OP
            and token.string == "="
            and self.lexed[at + 1].string in _AFTER_SPECIFIER
            for place in self._field(at, old)
        }

    def _field(self, end: int, old: str) -> list[Position]:
        """Where `old` is spelled in the expression of the field whose `=` is the token `end`.

        The expression starts at the bracket that no bracket after it closes.
        """
        depth = 0
        found: list[Position] = []
        for token in reversed(self.lexed[:end]):
            if token.type == tokenize.OP:
                depth += (token.string in _CLOSING) - (token.string in _OPENING)
            if depth < 0:
                break
            if token.type == tokenize.NAME and token.string == old:
                found.append(self._place(token.start))
        return found

    def _place(self, position: Position) -> Position:
        """`position` of a token, its column in characters, with the column in bytes."""
        line, column = position
        return line, len(self.lines[line - 1][:column].encode())

    def _placed(self, old: str) -> set[Position]:
        """Where the syntax tree places `old` as a name: a variable, a keyword or an attribute."""
        found: set[Position] = set()
        for node in ast.walk(self.tree):
            match node:
                case ast.Name(id=name) | ast.arg(arg=name) | ast.keyword(arg=name) if name == old:
                    found.add((node.lineno, node.col_offset))
                case ast.Attribute(attr=name) if name == old:
                    found.add(Spelling.tail(node))
        return found

    def _tokens(self, old: str) -> list[Token]:
        """Where the source spells `old` as an identifier, or as a string naming it.

        A string names it when it is `old` itself or a path ending in it (`pkg.mod.old`).
        """
        found: list[Token] = []
        for token in self.lexed:
            named = token.type == tokenize.NAME and token.string == old
            if named or (
                token.type == tokenize.STRING and Spelling._spells(token.string, old=old)
            ):
                found.append(Token(self._place(token.start), self._place(token.end), named))
        return found

    def covers(self, spans: Collection[Span], old: str, *, start: Position, end: Position) -> bool:
        """Whether `spans` and the syntax tree account for every spelling of `old` in a range.

        The range runs from `start` to `end`. A field printing its own source that spells `old`
        is never accounted for, whatever the spans: rewriting it changes the string.
        """
        starts = [(span.start_line, span.start_column) for span in spans]
        placed = self._placed(old)
        printed = {each for each in self._documented(old) if start <= each < end}
        return not printed and all(
            Spelling._accounted(token, starts, placed=placed)
            for token in self._tokens(old)
            if start <= token.start < end
        )
