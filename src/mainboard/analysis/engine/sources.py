# A file's bytes as an edit sees them: spans as byte ranges, in the file's own line ending.

import bisect
import re
from typing import TYPE_CHECKING

from patos import FrozenModel

if TYPE_CHECKING:
    from ..findings import Span


class TextEdit(FrozenModel):
    """Replace bytes `[start, end)` of one file with `text`.

    stacks: an append, which may share its point with other appends.
    """

    path: str
    start: int
    end: int
    text: bytes
    stacks: bool = False


class Source:
    """One file's bytes with line offsets, turning spans into byte ranges."""

    def __init__(self, data: bytes) -> None:
        self.data = data
        self.newline = b"\r\n" if b"\r\n" in data else b"\n"
        self.starts = [0, *(match.end() for match in re.finditer(b"\n", data))]
        if self.starts[-1] != len(data):
            self.starts.append(len(data))

    def encode(self, text: str) -> bytes:
        """`text` as this file's bytes, in its line ending."""
        return text.replace("\r\n", "\n").encode().replace(b"\n", self.newline)

    def indent(self, line: int) -> bytes:
        current = self.data[self.starts[line - 1] : self.starts[line]]
        return current[: len(current) - len(current.lstrip(b" \t"))]

    def line(self, offset: int) -> int:
        """The line holding byte `offset`."""
        return bisect.bisect_right(self.starts, offset)

    def lines(self, span: Span) -> tuple[int, int]:
        """Whole lines `span` covers, line ending included: what deleting or moving it removes."""
        return self.starts[span.start_line - 1], self.starts[(span.end_line or span.start_line)]

    def offset(self, line: int, *, column: int) -> int:
        return self.starts[line - 1] + column
