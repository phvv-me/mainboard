from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace

from pydantic import JsonValue

from mainboard.trials import (
    Admissibility,
    Cell,
    Declaration,
    Flag,
    Stance,
    Universe,
    Vocabulary,
    Word,
)

# The provenance a hermetic run stamps, standing in for a probe of real silicon so a receipt
# written under test carries the same shape on any machine that runs the suite.
PROBED = {
    "host": "bench",
    "card": "GPU-1111",
    "card_probed": "found",
    "card_name": "Test Card",
    "card_detail": "",
    "driver": "580.65.06",
    "runtime": "13.1",
    "capability": "sm_89",
    "commit": "abc1234def5678901234567890abcdef12345678",
    "tree": "fedcba9876543210fedcba9876543210fedcba98",
    "source_digest": "0f1e2d3c4b5a69788796a5b4c3d2e1f0",
    "worktree_dirty": False,
    "mirrored": False,
    "versions": {"polars": "1.0"},
}


class Taken:
    """A preflight that answers with fixed digests, so no test touches silicon or a repository.

    Patched in at the one seam a session reads its provenance through, which is the whole probe
    rather than the stamp alone, because admissibility is asked per lane after collection starts.

    stamped: the provenance every receipt of the run carries, the fixed probe when omitted.
    admissibility: what this tree is worth, admissible unless a test is about the other answer.
    """

    def __init__(
        self,
        root: Path,
        repo: Path,
        *,
        probed: Sequence[str] = (),
        stamped: Mapping[str, JsonValue] | None = None,
        admissibility: Admissibility = Admissibility.ADMISSIBLE,
    ) -> None:
        self.root = root
        self.repo = repo
        self.stamped = dict(stamped or PROBED)
        self.admissibility = admissibility

    @property
    def stamp(self) -> dict[str, JsonValue]:
        return dict(self.stamped)

    def admits(self, lane: Path) -> Admissibility:
        return self.admissibility

    def baselines(self, node: str) -> str:
        return f"baselines-of-{node or 'root'}"


class Item:
    """A collected trial, carrying exactly the four attributes `Trial` reads off a real one."""

    def __init__(self, nodeid: str, path: Path, params: dict[str, str] | None = None) -> None:
        self.nodeid = nodeid
        self.path = path
        self.user_properties: list[tuple[str, str]] = []
        self.callspec = SimpleNamespace(params=params) if params is not None else None


class Card:
    """One probed device, the five attributes provenance reads and nothing else."""

    def __init__(
        self,
        uuid: str = "GPU-1111",
        label: str = "Test Card",
        driver: str = "580.65.06",
        runtime: tuple[int, int] | None = (13, 1),
        arch_key: str = "sm_89",
    ) -> None:
        self.uuid = uuid
        self.label = label
        self.driver = driver
        self.runtime_version = runtime
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
