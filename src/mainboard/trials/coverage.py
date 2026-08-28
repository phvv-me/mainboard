# WHAT A COVERAGE QUESTION IS ASKED AT, AND WHAT ITS ANSWER LOOKS LIKE.
#
# AN EMPTY COORDINATE IS FOUR DIFFERENT FACTS AND THEY MUST NOT COLLAPSE. A host that genuinely
# carries no card, a probe that broke, a lane that names no subject and a run written before an
# axis existed all spelled the empty string, and each rule that produced that was right on its
# own: provenance records an absent device as empty, and the store normalises a null column to
# empty so an old run reads beside a new one. Together they erase the distinction that matters
# most to a measurement, which is whether the machine was known. So a cell carries the probe
# OUTCOME beside the probe VALUE, and a theory host and a broken probe are different cells.
#
# A CELL MAY OWE MORE THAN ONE SAMPLE. One passing receipt per key is the right completion rule
# for a claim that is either true or false on this machine, and the wrong one for a program whose
# subject is variance, where a cell owes N readings and a re-run must ADD to them rather than
# replay them. The target is declared per universe, the count is read across runs, and
# accumulation is what falls out: a partial cell is collected again and its new fragments join the
# ones already there.

from enum import StrEnum, auto

from patos import FrozenModel

# What a probe outcome column is called, one suffix on the axis it qualifies, so a store never has
# to be told which of its columns are outcomes and which are values.
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
    probing: axis to why it reads that, so an absent device and a broken probe stay apart.
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
        """How a status line spells this cell, its values, and any axis whose probe went wrong.

        A `found` or `unasked` axis with a value says everything by naming the value, and one
        without a value says everything by saying nothing. An `absent` or `failed` axis is the
        case a reader has to see, because the first means this run measured no such thing and the
        second means nobody knows what it measured.
        """
        parts = [value for value in self.values.values() if value]
        parts += [
            f"{axis} {outcome}"
            for axis, outcome in self.probing.items()
            if outcome in (Probed.ABSENT, Probed.FAILED) and not self.values.get(axis)
        ]
        return ", ".join(parts)


class LaneStatus(FrozenModel):
    """Whether one lane's data already exists at one cell, and what a run would still take.

    A lane is COMPLETE when the store already holds every sample its grid owes, PARTIAL when some
    are there, and MISSING when none are. Coverage is read across runs rather than inside one,
    because the question is whether the DATA exists; `run` names the newest run that contributed
    to it, which is the one a skip message cites.

    lane: the lane this answers for.
    want: how many receipts the lane's grid owes, its keys times the samples each cell owes.
    have: how many of them the store already holds.
    missing: the keys still short of their sample target, sorted.
    run: the newest run that contributed, empty when nothing did.
    cell: the coordinate this reading is scoped to.
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
        """`complete`, `partial` or `missing`, which is the whole of what a reader wants."""
        if self.want and not self.missing:
            return "complete"
        return "partial" if self.have else "missing"

    def line(self) -> str:
        """This lane's one status line, as a session prints it before anything runs."""
        where = f" from {self.run}" if self.run else ""
        short = ", ".join(self.missing[:3]) + ("..." if len(self.missing) > 3 else "")
        detail = f" missing {len(self.missing)}: {short}" if self.missing else where
        named = self.cell.named
        return (
            f"  {self.state:<8} {self.lane}{f' on {named}' if named else ''}  "
            f"{self.have}/{self.want}{detail}"
        )
