# WHERE A CONSUMER'S TRIALS LIVE, DECLARED ONCE AND READ BY EVERYTHING ELSE.
#
# A universe is a tree of nodes, one directory per claim, each holding its own receipt store. That
# is the only layout fact this subsystem holds, and it is a field rather than a constant so a
# consumer whose evidence lives somewhere else says so instead of moving its files.
#
# A NODE IS THE FIRST DIRECTORY UNDER THE ROOT, which is how a trial finds its own store without
# retyping anything: the folder a lane sits in IS the claim it serves. A flat universe, every lane
# in the root itself, is the same rule with an empty node and one store, so neither layout is a
# special case anywhere downstream.

from pathlib import Path

from patos import FrozenModel

from .dataset import Dataset
from .ledger import NESTED


class Universe(FrozenModel):
    """A consumer's trial tree: where its nodes are, how they store evidence, and what scopes it.

    root: the directory holding the nodes, a lane's own file living under one of them.
    evidence: the per-node path the receipt partitions sit under.
    axes: the coverage coordinates, each one a receipt column and each one asked of every lane.
        An axis is resolved from a trial's own parameters when it names one and from the run's
        probed provenance otherwise, so `model` comes off a parametrize grid and `card` off the
        machine without either being special-cased anywhere.
    probed: the logical packages whose provider distribution version every receipt records.
    nested: the receipt columns stored as JSON text rather than as parquet scalars.
    samples: how many passing receipts one cell owes before a lane is complete there. One suits a
        claim that is either true or false on this machine; a program whose subject is variance
        declares several and a re-run then accumulates toward the target instead of replaying it.
    """

    root: Path
    evidence: str = "evidence/receipts"
    axes: tuple[str, ...] = ()
    probed: tuple[str, ...] = ()
    nested: tuple[str, ...] = NESTED
    samples: int = 1

    @property
    def nodes(self) -> tuple[str, ...]:
        """Every node that has ever written a receipt, in name order.

        The flat layout answers with the empty node, which is the same store its lanes write to,
        so a caller sweeping a universe never has to ask which of the two shapes it is holding.
        """
        found = tuple(
            sorted(
                path.name
                for path in self.root.iterdir()
                if path.is_dir() and Dataset(path / self.evidence).parts
            )
            if self.root.is_dir()
            else ()
        )
        return found or (("",) if self.dataset("").parts else ())

    def dataset(self, node: str) -> Dataset:
        """One node's receipt store, read with this universe's declared axes."""
        return Dataset(
            self.root / node / self.evidence,
            axes=self.axes,
            nested=self.nested,
            node=node,
            samples=self.samples,
        )

    def node_of(self, path: Path) -> str:
        """Which node a file belongs to, the first directory under the root, empty when flat."""
        parts = path.resolve().parent.relative_to(self.root.resolve()).parts
        return parts[0] if parts else ""
