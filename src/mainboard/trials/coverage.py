# What a coverage question is asked at, and what its answer looks like.
#
# An empty coordinate is four different facts: a host with no card, a probe that broke, a lane that
# names no subject and a run written before the axis existed all read as the empty string. So a
# cell carries each axis's probe OUTCOME beside its VALUE, and a theory host and a broken probe are
# different cells. A cell may also owe several samples (a program whose subject is variance), read
# across runs, so a re-run of a partial cell adds fragments rather than replaying it.

from enum import StrEnum, auto

from patos import FrozenModel

# The suffix naming an axis's probe outcome column, so a store never has to be told which columns
# are outcomes and which are values.
PROBED = "_probed"


class Probed(StrEnum):
    """Why one axis of a cell reads what it reads, recorded beside the value itself."""

    FOUND = auto()
    ABSENT = auto()
    FAILED = auto()
    UNASKED = auto()


class Cell(FrozenModel):
    """One coordinate a coverage question is asked at, each axis's value beside its outcome.

    values: axis to what it reads, the empty string where it reads nothing.
    probing: axis to why it reads that.
    """

    values: dict[str, str] = {}
    probing: dict[str, Probed] = {}

    @property
    def filters(self) -> dict[str, str]:
        """Every column this cell pins, the axis values and their outcomes together."""
        return {
            **self.values,
            **{f"{axis}{PROBED}": str(outcome) for axis, outcome in self.probing.items()},
        }

    @property
    def key(self) -> tuple[tuple[str, str], ...]:
        """This cell as something hashable, for grouping lanes that share a coordinate."""
        return tuple(sorted(self.filters.items()))

    @property
    def named(self) -> str:
        """How a status line spells this cell: its values, then any empty `absent`/`failed` axis.

        Those two are the cases a reader must see: this run measured no such thing, or nobody
        knows what it measured.
        """
        parts = [value for value in self.values.values() if value]
        parts += [
            f"{axis} {outcome}"
            for axis, outcome in self.probing.items()
            if outcome in (Probed.ABSENT, Probed.FAILED) and not self.values.get(axis)
        ]
        return ", ".join(parts)


class LaneStatus(FrozenModel):
    """Whether one lane's data already exists at one cell, read across runs.

    want: the receipts the lane's grid owes, its keys times the samples each cell owes.
    have: how many of them the store already holds.
    missing: the keys still short of their sample target, sorted.
    run: the newest run that contributed, empty when nothing did; a skip message cites it.
    node: the claim whose store this was read from, empty for a flat universe.
    """

    lane: str
    want: int
    have: int
    missing: tuple[str, ...]
    run: str = ""
    cell: Cell = Cell()
    node: str = ""

    @property
    def state(self) -> str:
        """`complete` when every owed sample is stored, `partial` when some are, else `missing`."""
        if self.want and not self.missing:
            return "complete"
        return "partial" if self.have else "missing"

    def line(self) -> str:
        """This lane's status line, as a session prints it before anything runs."""
        where = f" from {self.run}" if self.run else ""
        short = ", ".join(self.missing[:3]) + ("..." if len(self.missing) > 3 else "")
        detail = f" missing {len(self.missing)}: {short}" if self.missing else where
        named = self.cell.named
        return (
            f"  {self.state:<8} {self.lane}{f' on {named}' if named else ''}  "
            f"{self.have}/{self.want}{detail}"
        )
