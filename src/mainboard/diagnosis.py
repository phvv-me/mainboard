# WHY A SETTLED RUN ENDED THE WAY IT DID, READ OFF THE OUTPUT IT LEFT BEHIND.
#
# A failed row used to say `failed` and its exit code while the output explaining it sat in the
# lake beside the run's receipts, so thirty two jobs dying on one loader error were thirty two
# identical rows and the line was found by hand hours later. The cause is now a column: the last
# meaningful line of the run's captured tail, which is what anyone would have quoted (a
# traceback's exception, the wrapper's kill stamp, a loader's missing symbol). Indented lines are
# skipped because traceback frames are indented and the exception is not; the exit trailer is
# skipped because the exit code is a column already. Only the tail is read: a training log is
# megabytes, a listing reads one per failed row, and the answer was never past a few hundred lines.

import re
from typing import TYPE_CHECKING

from .dispatch.evidence import printed
from .dispatch.state.captured import Captured
from .tracking import streamed

if TYPE_CHECKING:
    from .board import Board
    from .dispatch.state import RunRecord

# The tail is generous because a traceback is long; the cell is a cell.
TAIL = 200
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


def reason(board: Board, record: RunRecord) -> str:
    """Why `record` failed, from the output the sweep brought home, empty for a run that did not.

    Read here rather than remembered at settle time, so runs older than this column answer too
    and nothing is re-probed over ssh to fill a table.
    """
    stream, _ = streamed(record.name or "", handle=record.handle)
    kept = Captured(board.dispatcher.cache.session).transcript(stream, record.handle, last=TAIL)
    return cause(kept or "")
