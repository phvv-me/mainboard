# Naming a module's long literals: each lifted out of its function into a module constant.

import io
import re
import tokenize
from collections import Counter
from typing import TYPE_CHECKING

from ...edits import Edit, Insert, Replace
from ...findings import Span
from ..base import Operation
from ..model import Namer
from .imports import Statement

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

# What a constant's name may be once its underscore is added, and how much of a literal and of
# the code around it a question quotes.
_NAME = re.compile(r"_[A-Z][A-Z0-9_]{1,40}")
_QUOTED = 160
_AROUND = 2

_QUESTION = """\
Name module-level constants for the Python module `{path}`. Each numbered literal below is
written inside a function there and becomes a constant. Answer one name per literal, in order,
in UPPER_SNAKE_CASE without a leading underscore, saying what the value means where it is used
(SECONDS_PER_HOUR, NANOSECONDS_PER_SECOND, RECEIPT_TABLES_SQL), never what it is spelled as
(THREE_THOUSAND_SIX_HUNDRED). Two literals that mean different things get different names. Where
a literal lists names the project already gives it, reuse the one that means the same here.

{literals}
"""


class NameConstants(Operation):
    """Lift one module's long literals out of its functions into module constants.

    Each constant goes after the module's last top-level import, written as the literal was, and
    every occurrence of the same literal reads it. A literal is named by where it stands (the
    keyword, parameter or variable it is the whole value of) or else by a small model, which only
    ever chooses the name. A name that is not an upper-case identifier, that the module already
    spells, or that two meanings would share leaves its literal where it is.
    """

    @staticmethod
    def _asked(row: Mapping, literals: Sequence[Mapping], namer: Namer) -> list[str]:
        """The model's names for `literals`, each with an underscore, empty ones refused."""
        if not literals:
            return []
        lines = row["content"].splitlines()
        quoted = [
            f"{number}) a {each['kind']} literal `{each['text'][:_QUOTED]}`"
            + (
                f" (named elsewhere: {', '.join(given)})"
                if (given := namer.given.get(each["text"]))
                else ""
            )
            + " in:\n"
            + "\n".join(
                lines[max(each["start_line"] - 1 - _AROUND, 0) : each["start_line"] + _AROUND]
            )
            for number, each in enumerate(literals, 1)
        ]
        question = _QUESTION.format(path=row["path"], literals="\n\n".join(quoted))
        names = namer.names(question, len(literals)) or [""] * len(literals)
        return [f"_{name.upper().lstrip('_')}" if name else "" for name in names]

    @staticmethod
    def _chosen(row: Mapping) -> dict[str, str]:
        """Each literal's constant name by its text, a literal whose name cannot stand left out."""
        literals, namer = row["literals"], Namer()
        first = {
            text: next(e for e in literals if e["text"] == text)
            for text in dict.fromkeys(e["text"] for e in literals)
        }
        named = {
            text: f"_{each['naming'].upper()}" for text, each in first.items() if each["naming"]
        }
        unnamed = [each for text, each in first.items() if text not in named]
        asked = NameConstants._asked(row, unnamed, namer)
        names = named | dict(zip([each["text"] for each in unnamed], asked, strict=True))
        counts = Counter(names.values())
        chosen = {
            text: name
            for text, name in names.items()
            if counts[name] == 1
            and _NAME.fullmatch(name)
            and not re.search(rf"\b{name}\b", row["content"])
        }
        for text, name in chosen.items():
            namer.remember(text, name=name.removeprefix("_"))
        return chosen

    @staticmethod
    def _spelled(text: str) -> str:
        """`text` as a module constant's value, in parentheses when it must be.

        That is a value spanning lines as more than one string (`"a"` and `"b"` on two lines),
        which would otherwise read as an indented line.
        """
        strings = [
            token
            for token in tokenize.generate_tokens(io.StringIO(text).readline)
            if token.type == tokenize.STRING
        ]
        return f"({text})" if "\n" in text and len(strings) != 1 else text

    def edits(self, row: Mapping) -> list[Edit] | None:
        if not (chosen := NameConstants._chosen(row)):
            return None
        line = row["after_imports"]
        constants = "".join(
            f"{name} = {NameConstants._spelled(text)}\n" for text, name in chosen.items()
        )
        return [
            Insert(
                text="\n" + constants, after=Span(path=row["path"], start_line=line, end_line=line)
            ),
            *(
                Replace(span=Statement.spanned(each), text=chosen[each["text"]])
                for each in row["literals"]
                if each["text"] in chosen
            ),
        ]
