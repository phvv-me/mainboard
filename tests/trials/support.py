from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace

from mainboard.trials import Cell, Declaration, Flag, Stance, Universe, Vocabulary, Word

# The provenance a hermetic run stamps, standing in for a probe of real silicon so a receipt
# written under test carries the same shape on any machine that runs the suite.
PROBED = {
    "host": "bench",
    "card": "GPU-1111",
    "card_probed": "found",
    "card_name": "Test Card",
    "card_detail": "",
    "driver": "580.1",
    "capability": "sm_89",
    "commit": "abc1234",
    "worktree_dirty": False,
    "mirrored": False,
    "versions": {"polars": "1.0"},
}


class Item:
    """A collected trial, carrying exactly the four attributes `Trial` reads off a real one."""

    def __init__(self, nodeid: str, path: Path, params: dict[str, str] | None = None) -> None:
        self.nodeid = nodeid
        self.path = path
        self.user_properties: list[tuple[str, str]] = []
        self.callspec = SimpleNamespace(params=params) if params is not None else None


class Card:
    """One probed device, the four attributes provenance reads and nothing else."""

    def __init__(
        self,
        uuid: str = "GPU-1111",
        label: str = "Test Card",
        driver: tuple[int, int] | None = (580, 1),
        arch_key: str = "sm_89",
    ) -> None:
        self.uuid = uuid
        self.label = label
        self.driver_version = driver
        self.arch_key = arch_key


class Machine:
    """A probed machine answering with the cards it was handed, or raising as a broken probe."""

    def __init__(self, cards: Sequence[Card] = (), breaks: str = "") -> None:
        self.cards = tuple(cards)
        self.breaks = breaks

    @property
    def gpus(self) -> tuple[Card, ...]:
        if self.breaks:
            raise RuntimeError(self.breaks)
        return self.cards


def declaration(
    root: Path,
    *,
    axes: tuple[str, ...] = ("card", "model"),
    flags: tuple[Flag, ...] = (),
    repo: Path | None = None,
) -> Declaration:
    """A workspace declaration over `root`, with the axes and words the suite reads back."""
    return Declaration(
        universe=Universe(root=root, axes=axes, probed=("polars",)),
        words=Vocabulary(
            words=(
                Word(name="validated", letter="V", stance=Stance.CONFIRMS),
                Word(name="refuted", stance=Stance.REFUTES),
                Word(name="undecided"),
            )
        ),
        flags=flags,
        repo=repo,
    )


def cell(**values: str) -> Cell:
    """A cell whose every axis was found, which is what a probed run writes."""
    return Cell(values=values, probing=dict.fromkeys(values, "found"))
