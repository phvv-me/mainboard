from collections.abc import Sequence
from itertools import count
from pathlib import Path

import pytest

from mainboard.trials import (
    Dataset,
    Declaration,
    Figures,
    Fleet,
    Gap,
    LaneStatus,
    Local,
    Need,
    Refusal,
    Universe,
    rendered_twice,
)

from .support import cell, taken


class Drawn(Figures):
    """A render of one node's current view, deterministic unless told to stamp a counter."""

    def __init__(
        self, universe: Universe, needs: Sequence[Need], *, drifting: bool = False
    ) -> None:
        super().__init__(universe)
        self.declared = tuple(needs)
        self.ticks = count()
        self.drifting = drifting

    @property
    def needs(self) -> tuple[Need, ...]:
        return self.declared

    def draw(self, out: Path) -> tuple[Path, ...]:
        keys = [str(row["key"]) for row in self.rows("alpha")]
        drift = f" {next(self.ticks)}" if self.drifting else ""
        (out / "figure.txt").write_text(",".join(sorted(keys)) + drift)
        return ((out / "figure.txt"),)


def status(node: str, lane: str, **values: str) -> LaneStatus:
    return LaneStatus(lane=lane, want=1, have=0, missing=("a",), cell=cell(**values), node=node)


def test_a_render_refuses_with_every_gap_named_rather_than_drawing_a_partial_view(
    store: Dataset, declared: Declaration
) -> None:
    figures = Drawn(
        declared.universe,
        (
            Need(node="ghost", lane="l"),
            Need(node="alpha", lane="l", keys=("a", "b"), least=2),
        ),
    )
    gaps = figures.gaps()
    assert [gap.why for gap in gaps] == [
        "no such node in the universe",
        "0 current trials, the render draws 2",
        "no current receipt at this key",
        "no current receipt at this key",
    ]
    with pytest.raises(Refusal, match="the render REFUSES: 4 receipt"):
        figures.render(store.root.parent / "out")

    taken(store, "run-1", {"lane": "l", "key": "a"}, {"lane": "l", "key": "b"})
    settled = Drawn(declared.universe, (Need(node="alpha", lane="l", keys=("a", "b"), least=2),))
    assert settled.gaps() == ()


def test_a_gap_names_what_was_wanted_and_where() -> None:
    assert Gap(node="alpha", lane="l", key="a", why="gone").line() == "  alpha l[a]: gone"
    assert Gap(node="", lane="l", key="", why="gone").line() == "  . l: gone"


def test_a_render_lands_in_a_clean_directory_and_two_runs_of_it_must_match_byte_for_byte(
    store: Dataset, declared: Declaration, tmp_path: Path
) -> None:
    taken(store, "run-1", {"lane": "l", "key": "a"}, {"lane": "other", "key": "b"})
    figures = Drawn(declared.universe, (Need(node="alpha", lane="l"),))
    out = tmp_path / "out"
    (out / "stale").mkdir(parents=True)
    written = figures.render(out)
    assert written == (out / "figure.txt",)
    assert not (out / "stale").exists()
    assert (out / "figure.txt").read_text() == "a,b"

    assert rendered_twice(figures, tmp_path / "check") == ()
    assert rendered_twice(
        Drawn(declared.universe, (Need(node="alpha", lane="l"),), drifting=True),
        tmp_path / "drift",
    ) == ("figure.txt",)


def test_a_render_reads_every_sample_when_the_cells_owe_more_than_one(
    store: Dataset, declared: Declaration
) -> None:
    where = cell(card="GPU-1", model="qwen").filters
    taken(store, "run-1", {"lane": "l", "key": "a", **where})
    taken(store, "run-2", {"lane": "l", "key": "a", **where})
    figures = Drawn(declared.universe, ())
    assert len(figures.rows("alpha")) == 1
    assert len(figures.rows("alpha", every=True)) == 2
    assert figures.rows("alpha", lane="nothing-like-this") == []
    assert figures.rows("beta") == []


def test_running_here_is_one_partition_and_nothing_dispatched() -> None:
    lanes = (status("alpha", "l"), status("beta", "l"), status("beta", "m"))
    partitions = Local().partitions(lanes)
    assert len(partitions) == 1
    assert partitions[0].lanes == ("l", "m") and partitions[0].name == "root"
    assert Local().dispatch(partitions[0]) == ""


def test_the_hermetic_boundary_is_one_claim_and_one_coordinate_per_process() -> None:
    lanes = (
        status("alpha", "l", card="GPU-1"),
        status("alpha", "m", card="GPU-1"),
        status("alpha", "l", card="GPU-2"),
        status("beta", "n", card="GPU-1"),
    )
    partitions = Fleet().partitions(lanes)
    assert [(one.node, one.lanes, one.name) for one in partitions] == [
        ("alpha", ("l", "m"), "alpha-GPU-1"),
        ("alpha", ("l",), "alpha-GPU-2"),
        ("beta", ("n",), "beta-GPU-1"),
    ]
    with pytest.raises(NotImplementedError, match="assigned by UUID"):
        Fleet().dispatch(partitions[0])
