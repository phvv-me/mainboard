# Where a consumer's trials live. A universe is a tree of nodes, one directory per claim, each with
# its own receipt store. A node is the first directory under the root, so the folder a lane sits
# in IS the claim it serves; a flat universe is the same rule with the empty node and one store.

from pathlib import Path
from typing import ClassVar

from patos import FrozenModel

from .dataset import Dataset
from .ledger import NESTED


class Universe(FrozenModel):
    """A consumer's trial tree: where its nodes are, how they store evidence, and what scopes it.

    root: the directory holding the nodes, a lane's own file living under one of them.
    evidence: the per-node path the receipt partitions sit under.
    axes: the coverage coordinates, each a receipt column asked of every lane. An axis resolves
        from a trial's own parameter of that name, else from a lane marker of that name
        (`@pytest.mark.phase("2")`), else from the run's probed provenance (`card`).
    probed: the logical packages whose provider distribution version every receipt records.
    nested: the receipt columns stored as JSON text rather than as parquet scalars.
    samples: how many passing receipts one cell owes before a lane is complete there. A program
        whose subject is variance declares several, and a re-run then accumulates toward them.
    """

    root: Path
    datasets: Path | None = None
    dataset_type: ClassVar[type[Dataset]] = Dataset
    evidence: str = "evidence/receipts"
    axes: tuple[str, ...] = ()
    probed: tuple[str, ...] = ()
    nested: tuple[str, ...] = NESTED
    samples: int = 1

    @property
    def storage_root(self) -> Path:
        """The declared data root, or the conventional sibling of an experiments tree."""
        if self.datasets is not None:
            return self.datasets
        if self.root.name == "experiments":
            return self.root.parent / "datasets" / "experiments"
        return self.root

    @property
    def nodes(self) -> tuple[str, ...]:
        """Every node that has ever written a receipt, in name order; the empty node when flat."""
        found = tuple(
            sorted(
                path.name
                for path in self.storage_root.iterdir()
                if path.is_dir() and Dataset(path / self.evidence).parts
            )
            if self.storage_root.is_dir()
            else ()
        )
        return found or (("",) if self.dataset("").parts else ())

    def dataset(self, node: str) -> Dataset:
        """One node's receipt store, read with this universe's declared axes."""
        return self.dataset_type(
            self.storage_root / node / self.evidence,
            axes=self.axes,
            nested=self.nested,
            node=node,
            samples=self.samples,
        )

    def node_of(self, path: Path) -> str:
        """Which node a file belongs to, the first directory under the root, empty when flat."""
        parts = path.resolve().parent.relative_to(self.root.resolve()).parts
        return parts[0] if parts else ""
