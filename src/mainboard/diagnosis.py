# WHY A SETTLED RUN ENDED THE WAY IT DID, READ OFF THE OUTPUT IT LEFT BEHIND.
#
# A failed row used to say `failed` and its exit code, and nothing else. The output that says
# what actually happened was on disk the whole time, beside the run's own receipts where the
# sweep put it, and reading it meant knowing that, finding the handle's log and opening it. So
# thirty two jobs that all died on one line of a loader error were thirty two identical `failed`
# rows, and the line itself was found by hand hours later.
#
# The cause is therefore a column: the last meaningful line of the run's own captured tail,
# which for the failures a dispatch actually produces is the line anyone would have quoted
# anyway. A python traceback ends with its exception, a killed job ends with the wrapper's own
# stamp, and a loader failure ends with the symbol it could not find. Indented lines are skipped
# because a traceback's frames are indented and its exception is not, and the wrapper's exit
# trailer is skipped because the exit code is already a column of its own.
#
# Only the tail is read. A training loop's log is megabytes, a listing reads one per failed row,
# and nothing beyond the last few kilobytes has ever been the answer.

import re
from typing import TYPE_CHECKING

from .batch.runner import directory
from .dispatch.evidence import printed
from .tracking import streamed

if TYPE_CHECKING:
    from pathlib import Path

    from .board import Board
    from .dispatch.state import RunRecord

# How much of the end of a log is read, and how wide the answer is allowed to be. The tail is
# generous because a line can be long; the cell is a cell.
TAIL_BYTES = 8192
WIDTH = 240

# The wrapper's own closing stamp, which says the same thing the exit code column already says.
_TRAILER = re.compile(r"^exit=-?\d+$")


def cause(log: str) -> str:
    """The one line of `log` that says why the run ended, empty when it says nothing.

    The last unindented line that is neither blank nor the wrapper's exit stamp, which is a
    traceback's exception, a loader's missing symbol, or a scheduler's own kill notice. The
    receipts frame a rented run closes its log with is the wrapper's too, and never a cause.
    """
    for line in reversed(printed(log).splitlines()):
        stripped = line.strip()
        if not stripped or line[:1].isspace() or _TRAILER.match(stripped):
            continue
        return stripped[:WIDTH]
    return ""


def transcript(path: Path) -> str:
    """The last `TAIL_BYTES` of `path` as text, empty when there is no such file to read.

    Bytes rather than lines, so a log of any size costs one seek, and the partial character a
    cut can land in the middle of is dropped rather than raising.
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

    Read here rather than remembered at settle time so a run that ended before this column
    existed answers too, and so nothing has to be re-probed over ssh to fill a table in.
    """
    return cause(transcript(stored_log(board, record)))
