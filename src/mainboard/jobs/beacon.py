# What a dispatched test job says about its own progress, in lines a waiter reads off its log.
#
# Agents waiting on a job wrote their own polling loops, 56 of them in seven sessions: tail the
# log, count pytest's dots, guess from `nvidia-smi` whether anything was still happening. The
# dots say nothing a reader can count, since `-q` prints no names and a fresh-process lane is a
# new pytest session per cell. So the runner reports through its own channel instead: how many
# cells the job holds, each cell's outcome as it lands, and the moment the pytest session ended,
# which is what lets a job whose process lingers past its tests be settled on what they said.
#
# Three marker lines, written by the runner's pytest plugin (`jobs/pytest.py`) and by a
# fresh-process lane, only in a dispatched job, where the log is read by a machine; a run at this
# workstation's own terminal keeps pytest's output as it always was:
#
#     mainboard-cells: 12
#     mainboard-cell: passed test_lane.py::test_cell[gpt2]
#     mainboard-session: 0
#
# A marker may land in the middle of pytest's progress dots, so it is read wherever it starts on
# a line and runs to the line's end, and `unbeaconed` removes exactly that span, which gives the
# reader of `logs` pytest's own output back byte for byte.

import os
import re

from patos import FrozenModel

# The three markers, each followed by one space and its value.
CELLS = "mainboard-cells:"
CELL = "mainboard-cell:"
SESSION = "mainboard-session:"

# Set in the environment of each process a fresh-process lane runs a cell in. The lane declares
# the total and ends the session itself, so a child reports its one cell and nothing else.
NESTED = "MAINBOARD_FRESH_CELL"

# Every marker span in a log, from the marker to the end of its line.
_SPAN = re.compile(rf"(?:{CELLS}|{CELL}|{SESSION}) [^\n]*\n?")

# The outcome a cell settles on when several of its phases reported, worst first.
_WORST = ("failed", "skipped", "passed")


def say(marker: str, value: str) -> None:
    """Write one marker line straight to the job's output descriptor.

    The descriptor rather than `sys.stdout`: pytest swaps `sys.stdout` for its capture while a
    test runs, and a marker must reach the log whatever pytest is capturing at that moment.
    """
    os.write(1, f"{marker} {value}\n".encode())


class Progress(FrozenModel):
    """What a job's markers say so far.

    total: the cells the job declared, None before it declared any.
    cells: each reported cell's settled outcome, in the order the cells first landed.
    session: the pytest session's exit status once it ended, None while it runs.
    """

    total: int | None = None
    cells: tuple[tuple[str, str], ...] = ()
    session: int | None = None

    @classmethod
    def read(cls, log: str) -> Progress:
        """The progress `log`'s markers add up to; empty for a log that carries none.

        log: the job's captured output so far.
        """
        total: int | None = None
        session: int | None = None
        outcomes: dict[str, str] = {}
        for span in _SPAN.findall(log):
            marker, _, value = span.strip().partition(" ")
            if marker == CELL:
                outcome, _, cell = value.partition(" ")
                held = outcomes.get(cell, outcome)
                outcomes[cell] = min(held, outcome, key=_rank)
            elif marker == CELLS and total is None:
                total = _number(value)
            elif marker == SESSION:
                session = _number(value)
        return cls(total=total, cells=tuple(outcomes.items()), session=session)

    @property
    def done(self) -> int:
        """How many cells have landed."""
        return len(self.cells)

    @property
    def failed(self) -> int:
        """How many landed cells failed."""
        return sum(outcome == "failed" for _, outcome in self.cells)

    @property
    def counted(self) -> str:
        """`done/total`, the total a question mark until declared, empty with nothing to count."""
        if self.total is None and not self.cells:
            return ""
        return f"{self.done}/{'?' if self.total is None else self.total}"


def unbeaconed(log: str) -> str:
    """`log` with every marker span removed, pytest's own output left exactly as it printed.

    log: the job's captured output.
    """
    return _SPAN.sub("", log)


def _rank(outcome: str) -> int:
    """How bad an outcome is, a word pytest never uses ranked as a pass."""
    return _WORST.index(outcome) if outcome in _WORST else len(_WORST)


def _number(value: str) -> int | None:
    """A marker's integer value, None for one torn mid-write."""
    try:
        return int(value)
    except ValueError:
        return None
