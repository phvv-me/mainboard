# TOML's renderings of the structural operations: a manifest's tables and arrays, edited by line.

import re
from typing import TYPE_CHECKING

from ..edits import Delete, Edit, Replace
from ..findings import Span
from .base import Operation

if TYPE_CHECKING:
    from collections.abc import Mapping


class DropExclude(Operation):
    """Take a pattern out of a rule's `exclude` array, the line it stood on along with it.

    A pattern alone on its line goes with the line; one inside a one-line array is cut out of the
    array. Any other spelling is refused rather than guessed at.
    """

    def edits(self, row: Mapping) -> list[Edit] | None:
        lines = row["content"].splitlines(keepends=True)
        text = lines[row["start_line"] - 1]
        quoted = rf"(?:\"{re.escape(row['pattern'])}\"|'{re.escape(row['pattern'])}')"
        span = Span(path=row["path"], start_line=row["start_line"], end_line=row["start_line"])
        if re.fullmatch(rf"\s*{quoted}\s*,?\s*(#.*)?", text.rstrip("\r\n")):
            return [Delete(span=span)]
        cut = re.sub(rf"{quoted}\s*,\s*|,?\s*{quoted}(?=\s*\])", "", text, count=1)
        if cut == text or "[" not in text:
            return None
        line = span.model_copy(update={"end_line": row["start_line"] + 1})
        return [Replace(span=line, text=cut)]
