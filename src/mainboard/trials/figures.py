# The figure contract, which is all of the figure this subsystem owns.
#
# A render reads receipts only, declares every receipt it draws from, and refuses with every gap
# named rather than draw a partial view: a table that quietly lost its worst row reads as good
# news. A render is also deterministic, a gate rather than a nicety: two renders over the same
# receipts must be byte-identical so a rerun diffs empty. `rendered_twice` checks it; passing it is
# the consumer's discipline (no timestamps, explicit orders, fixed float precision, no wall-clock
# metadata). The plotting itself stays with the consumer.

import abc
from functools import partial
from shutil import rmtree
from typing import TYPE_CHECKING

from patos import FrozenModel

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from pydantic import JsonValue

    from .universe import Universe


class Need(FrozenModel):
    """One receipt set a render draws from, declared so a missing one is named and not skipped.

    lane: a fragment of the lane's own id, every lane of the node when empty.
    keys: the keys that must all be present.
    least: how many CELLS the lane must have, the distinct coordinates a figure draws a line
        through; not the universe's samples per cell. A render wanting the repeats reads every
        sample instead.
    """

    node: str
    lane: str = ""
    keys: tuple[str, ...] = ()
    least: int = 1


class Gap(FrozenModel):
    """One declared receipt a render wanted and could not read."""

    node: str
    lane: str
    key: str
    why: str

    def line(self) -> str:
        """This gap as the refusal prints it."""
        where = f"{self.node or '.'} {self.lane}" + (f"[{self.key}]" if self.key else "")
        return f"  {where}: {self.why}"


class Refusal(RuntimeError):
    """Every gap at once, because a reader fixing one missing lane wants to see all of them."""

    def __init__(self, gaps: Sequence[Gap]) -> None:
        self.gaps = tuple(gaps)
        listing = "\n".join(gap.line() for gap in self.gaps)
        super().__init__(
            f"the render REFUSES: {len(self.gaps)} receipt(s) it draws from are missing.\n"
            f"{listing}\n"
            "Take the named lanes and render again; nothing was written."
        )


class Figures(abc.ABC):
    """A render that runs off one universe's receipts and refuses on a receipt it cannot read."""

    def __init__(self, universe: Universe) -> None:
        self.universe = universe

    @property
    @abc.abstractmethod
    def needs(self) -> tuple[Need, ...]:
        """Every receipt set this render draws from, checked before a single file is written."""

    @abc.abstractmethod
    def draw(self, out: Path) -> tuple[Path, ...]:
        """Write every artifact under `out` and return what was written."""

    def gaps(self) -> tuple[Gap, ...]:
        """Every declared need this universe cannot satisfy, in declaration order."""
        found: list[Gap] = []
        for need in self.needs:
            gap = partial(Gap, node=need.node, lane=need.lane)
            if not (self.universe.root / need.node).is_dir():
                found.append(gap(key="", why="no such node in the universe"))
                continue
            rows = self.rows(need.node, lane=need.lane)
            if len(rows) < need.least:
                why = f"{len(rows)} current trials, the render draws {need.least}"
                found.append(gap(key="", why=why))
            present = {str(row.get("key", "")) for row in rows}
            found.extend(
                gap(key=key, why="no current receipt at this key")
                for key in need.keys
                if key not in present
            )
        return tuple(found)

    def render(self, out: Path) -> tuple[Path, ...]:
        """Check every declared need, then write the whole render into an emptied `out`.

        Emptying first means a rerun cannot leave an artifact the render no longer draws.
        """
        gaps = self.gaps()
        if gaps:
            raise Refusal(gaps)
        if out.exists():
            rmtree(out)
        out.mkdir(parents=True)
        return self.draw(out)

    def rows(
        self, node: str, *, lane: str = "", every: bool = False
    ) -> list[dict[str, JsonValue]]:
        """One node's passing receipts as plain records, their JSON columns decoded.

        lane: a fragment of the lane id, every lane when empty.
        every: every sample rather than the newest of each cell, for a repeated-sample program;
            a representative table must not have it.
        """
        store = self.universe.dataset(node)
        return [
            store.decoded(row)
            for row in store.passing(every=every).to_dicts()
            if not lane or lane in str(row.get("lane", ""))
        ]


def rendered_twice(figures: Figures, under: Path) -> tuple[str, ...]:
    """Render twice under `under` and name every artifact whose bytes differ or that one lacks.

    Empty is the gate passing; any name means the render carries something not in the receipts.
    """
    written = {
        side: {
            path.relative_to(under / side).as_posix(): path.read_bytes()
            for path in figures.render(under / side)
        }
        for side in ("first", "second")
    }
    names = set(written["first"]) | set(written["second"])
    return tuple(
        sorted(name for name in names if written["first"].get(name) != written["second"].get(name))
    )
