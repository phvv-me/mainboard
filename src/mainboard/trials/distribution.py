# HOW A COLLECTED TRIAL SET IS SPREAD OVER MACHINES, AS AN INTERFACE AND ONE IMPLEMENTATION.
#
# MULTI-GPU IS MULTI-JOB HERE AND IS NEVER IN-PROCESS. A collective changes the arithmetic under
# measurement: it reorders reductions, it introduces a second allocator's fragmentation into the
# first one's readings, and it makes the number a trial reports a fact about the world size rather
# than about the operation. A subsystem whose whole output is measurements cannot buy throughput
# with that, so there is no device mesh, no spawn and no rank here, and there never will be. A run
# that wants four cards runs four processes, each measuring one card, each writing its own
# fragments, and the store joins them afterwards because the run rides as a column.
#
# THE FIRST REAL IMPLEMENTATION IS THE HERMETIC UNIVERSE EXECUTOR, and this seam is shaped for it
# rather than for a general scheduler. Its contract, stated here so it lands on a seam that fits:
# one partition is ONE CLAIM DIRECTORY, run by ONE FRESH PYTEST PROCESS, holding ONE GPU assigned
# by UUID, and that process EXITS before the next partition starts. Nothing is shared across the
# boundary, which is what makes it hermetic: a session fixture cannot outlive its own claim, a
# process-global knob cannot reach the claim collected after it, and an allocator cannot hand one
# universe the fragmentation another left. The receipts are settled from the fragments the process
# left on disk, not from anything it returned, so a partition that was killed still contributes
# every trial it took. That is `Fleet` below, and it is deliberately not built in this pass.
#
# THE PRIOR ART FOR THE CONTRACT IS DataJoint's AutoPopulate. Its `key_source` declares where the
# work comes from, `populate` computes the keys that source implies and are not yet in the table,
# and it runs exactly those, so the missing set is DERIVED from the data rather than tracked beside
# it. That is the same shape as `Dataset.status` and the partitions below: coverage says which
# cells are short, and dispatch runs those and only those. Naming it here so the seam is built
# against a design that has been load bearing for a decade rather than against a fresh guess.
#
# `Local` is what runs today: one partition, this process, nothing dispatched.

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from .coverage import Cell

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .coverage import LaneStatus


@dataclass(frozen=True, slots=True)
class Partition:
    """One slice of a collected trial set that can run as a process of its own.

    node: the claim whose lanes this holds.
    cell: the coordinate every lane in it shares, which is what the process must be pinned to.
    lanes: the lane ids, in collection order.
    """

    node: str
    cell: Cell
    lanes: tuple[str, ...]

    @property
    def name(self) -> str:
        """This partition's handle-friendly name, the claim and the values it is pinned to.

        The axis VALUES alone, never their probe outcomes, since a name is read by a person and
        `alpha-GPU-1` says what `alpha-GPU-1-found` does with less of it.
        """
        values = [value for value in self.cell.values.values() if value]
        return "-".join([self.node or "root", *values])


class Distribution(Protocol):
    """How a collected trial set becomes processes, and what running one of them means."""

    def dispatch(self, partition: Partition) -> str:
        """Start `partition` and return the handle it can be settled through, empty for here."""

    def partitions(self, lanes: Sequence[LaneStatus]) -> tuple[Partition, ...]:
        """Split what was collected into the units that may run independently."""


class Local:
    """Everything collected runs in this process, which is what a plain session already does."""

    def dispatch(self, partition: Partition) -> str:
        """Nothing to start, because the caller is already the process that runs it."""
        return ""

    def partitions(self, lanes: Sequence[LaneStatus]) -> tuple[Partition, ...]:
        """One partition holding every collected lane, since one process takes them all."""
        return (
            Partition(
                node="", cell=Cell(), lanes=tuple(dict.fromkeys(status.lane for status in lanes))
            ),
        )


class Fleet:
    """One claim per fresh process on one assigned card, settled from the fragments it left.

    Partitioning is real here and dispatch is not: splitting by claim and coordinate is a pure
    function of what was collected and is the half a caller can already inspect, plan and test,
    while starting the process is the half that needs a card assignment, a job wrapper and a
    settle path, and those land with the executor rather than ahead of it.
    """

    def dispatch(self, partition: Partition) -> str:
        """Refuse, naming the contract the executor has to satisfy before this can answer."""
        raise NotImplementedError(
            f"the hermetic universe executor is not built yet, so {partition.name} cannot be "
            "dispatched. Its contract is one claim directory, one fresh pytest process, one GPU "
            "assigned by UUID, process exit before the next partition, and receipts settled from "
            "the fragments that process left on disk"
        )

    def partitions(self, lanes: Sequence[LaneStatus]) -> tuple[Partition, ...]:
        """One partition per claim and coordinate, which is the hermetic boundary.

        A claim is the isolation unit because a session fixture is the physical acquisition
        unit: a whole campaign's checkpoint, activations and warmed card belong to one claim and
        must die with it. The coordinate splits beside it because two cards are two processes.
        """
        grouped: dict[tuple[str, tuple[tuple[str, str], ...]], list[str]] = {}
        cells: dict[tuple[tuple[str, str], ...], Cell] = {}
        for status in lanes:
            cells[status.cell.key] = status.cell
            grouped.setdefault((status.node, status.cell.key), []).append(status.lane)
        return tuple(
            Partition(node=node, cell=cells[at], lanes=tuple(dict.fromkeys(found)))
            for (node, at), found in sorted(grouped.items())
        )
