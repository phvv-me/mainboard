# How a collected trial set is spread over machines.
#
# Multi-GPU is multi-job, never in-process: a collective reorders reductions, adds a second
# allocator's fragmentation and makes a reading a fact about the world size, so there is no mesh,
# spawn or rank here. A run wanting four cards runs four processes, each writing its own fragments,
# and the store joins them because the run rides as a column.
#
# `Fleet` is shaped for the hermetic universe executor, not yet built: one partition is one claim
# directory, run by one fresh pytest process holding one GPU assigned by UUID, which exits before
# the next starts, so no fixture, knob or fragmentation crosses the boundary. Receipts settle from
# the fragments left on disk, so a killed partition still contributes every trial it took. The
# prior art is DataJoint's AutoPopulate: the missing set is derived from the data
# (`Dataset.status`) and dispatch runs exactly that. `Local` is what runs today.

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from .coverage import Cell

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .coverage import LaneStatus


@dataclass(frozen=True, slots=True)
class Partition:
    """One slice of a collected trial set that can run as a process of its own.

    cell: the coordinate every lane in it shares, which the process must be pinned to.
    lanes: the lane ids, in collection order.
    """

    node: str
    cell: Cell
    lanes: tuple[str, ...]

    @property
    def name(self) -> str:
        """A job-handle name: the claim and the axis values it is pinned to, never the outcomes."""
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
        return ""

    def partitions(self, lanes: Sequence[LaneStatus]) -> tuple[Partition, ...]:
        return (
            Partition(
                node="", cell=Cell(), lanes=tuple(dict.fromkeys(status.lane for status in lanes))
            ),
        )


class Fleet:
    """One claim per fresh process on one assigned card, settled from the fragments it left.

    Partitioning is real, a pure function of what was collected; dispatch lands with the executor.
    """

    def dispatch(self, partition: Partition) -> str:
        raise NotImplementedError(
            f"the hermetic universe executor is not built yet, so {partition.name} cannot be "
            "dispatched. Its contract is one claim directory, one fresh pytest process, one GPU "
            "assigned by UUID, process exit before the next partition, and receipts settled from "
            "the fragments that process left on disk"
        )

    def partitions(self, lanes: Sequence[LaneStatus]) -> tuple[Partition, ...]:
        """One partition per claim and coordinate, the hermetic boundary.

        A claim is the isolation unit because a campaign's checkpoint and warmed card belong to one
        claim and must die with it; two cards are two processes.
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
