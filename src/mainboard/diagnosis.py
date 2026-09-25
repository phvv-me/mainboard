# WHY A SETTLED RUN ENDED THE WAY IT DID, READ OFF THE OUTPUT IT LEFT BEHIND.
#
# A failed row used to say `failed` and its exit code while the output explaining it sat on disk
# beside the run's receipts, so thirty two jobs dying on one loader error were thirty two
# identical rows and the line was found by hand hours later. The cause is now a column: the last
# meaningful line of the run's captured tail, which is what anyone would have quoted (a
# traceback's exception, the wrapper's kill stamp, a loader's missing symbol). Indented lines are
# skipped because traceback frames are indented and the exception is not; the exit trailer is
# skipped because the exit code is a column already. Only the tail is read: a training log is
# megabytes, a listing reads one per failed row, and the answer was never beyond a few kilobytes.

import re
from typing import TYPE_CHECKING

from .batch.runner import directory
from .dispatch.evidence import printed
from .tracking import streamed

if TYPE_CHECKING:
    from pathlib import Path

    from .board import Board
    from .dispatch.state import RunRecord

# The tail is generous because a line can be long; the cell is a cell.
TAIL_BYTES = 8192
WIDTH = 240

# The wrapper's own closing stamp.
_TRAILER = re.compile(r"^exit=-?\d+$")


def cause(log: str) -> str:
    """The last unindented line of `log` that is neither blank nor the exit stamp, else empty.

    The receipts frame a rented run closes its log with is the wrapper's too, never a cause.
    """
    for line in reversed(printed(log).splitlines()):
        stripped = line.strip()
        if not stripped or line[:1].isspace() or _TRAILER.match(stripped):
            continue
        return stripped[:WIDTH]
    return ""


def transcript(path: Path) -> str:
    """The last `TAIL_BYTES` of `path` as text, empty when there is no such file to read.

    Bytes rather than lines, so any log costs one seek; a character cut in half is replaced
    rather than raising.
    """
    try:
        with path.open("rb") as opened:
            opened.seek(0, 2)
            opened.seek(max(0, opened.tell() - TAIL_BYTES))
            return opened.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def stored_log(board: Board, record: RunRecord) -> Path:
    """Where the sweep left `record`'s captured output, whether or not it has landed yet."""
    stream, _ = streamed(record.name or "", handle=record.handle)
    return directory(board, stream) / f"{record.handle}.log"


def reason(board: Board, record: RunRecord) -> str:
    """Why `record` failed, from the log the sweep brought home, empty for a run that did not.

    Read here rather than remembered at settle time, so runs older than this column answer too
    and nothing is re-probed over ssh to fill a table.
    """
    return cause(transcript(stored_log(board, record)))
