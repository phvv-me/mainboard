# THE FIGURE CONTRACT, WHICH IS ALL OF THE FIGURE THIS SUBSYSTEM OWNS.
#
# A render reads receipts and nothing else, DECLARES every receipt it renders from, and REFUSES
# with every gap named rather than drawing a partial view. A partial figure is worse than no
# figure, because a table that quietly lost its worst row reads as a table whose worst row is good
# news, and nothing on the page says otherwise.
#
# AND A RENDER IS DETERMINISTIC, WHICH IS A GATE RATHER THAN A NICETY. Two runs of the same render
# over the same receipts must produce byte-identical files, so a rerun can be diffed against the
# last one and come back empty. `rendered_twice` is that check, and a consumer wires it into its
# own suite; what makes it pass is the consumer's own discipline, no timestamps, explicit orders,
# fixed float precision and no library metadata stamped with the wall clock.
#
# THE PLOTTING STAYS WITH THE CONSUMER. A figure is a claim about a specific measurement and no
# generic base can draw one. What is here is the refusal, the current-view read behind it and the
# determinism check, which is the part every consumer would otherwise write again and slightly
# differently.

import abc
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

    node: the claim whose store holds it.
    lane: a fragment of the lane's own id, every lane of the node when empty.
    keys: the keys that must all be present.
    least: how many CELLS the lane must have at minimum, which is a different question from the
        universe's samples per cell. This counts distinct coordinates a figure needs to draw a
        line through; that counts repeated readings one coordinate owes before it is settled. A
        render wanting the repeats reads every sample rather than raising this number.
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
        """This gap as the refusal prints it, naming what was wanted and where."""
        where = f"{self.node or '.'} {self.lane}" + (f"[{self.key}]" if self.key else "")
        return f"  {where}: {self.why}"


class Refusal(RuntimeError):
    """Every gap at once, because a reader fixing one missing lane wants to see all of them."""

    def __init__(self, gaps: Sequence[Gap]) -> None:
        """gaps: every receipt the render wanted and could not read."""
        self.gaps = tuple(gaps)
        listing = "\n".join(gap.line() for gap in self.gaps)
        super().__init__(
            f"the render REFUSES: {len(self.gaps)} receipt(s) it draws from are missing.\n"
            f"{listing}\n"
            "Take the named lanes and render again; nothing was written."
        )


class Figures(abc.ABC):
    """A render that runs off one universe's receipts and refuses on a receipt it cannot read.

    universe: the trial tree whose current view this draws from.
    """

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
        """Every declared need this universe cannot satisfy, in the order they were declared."""
        found: list[Gap] = []
        for need in self.needs:
            if not (self.universe.root / need.node).is_dir():
                found.append(
                    Gap(node=need.node, lane=need.lane, key="", why="no such node in the universe")
                )
                continue
            rows = self.rows(need.node, lane=need.lane)
            if len(rows) < need.least:
                found.append(
                    Gap(
                        node=need.node,
                        lane=need.lane,
                        key="",
                        why=f"{len(rows)} current trials, the render draws {need.least}",
                    )
                )
            present = {str(row.get("key", "")) for row in rows}
            found.extend(
                Gap(node=need.node, lane=need.lane, key=key, why="no current receipt at this key")
                for key in need.keys
                if key not in present
            )
        return tuple(found)

    def render(self, out: Path) -> tuple[Path, ...]:
        """Check every declared need, then write the whole render into a clean `out`.

        The directory is emptied first so a rerun cannot leave an artifact from a render that no
        longer draws it, which is what makes two runs comparable byte for byte.
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
        """One node's passing receipts as plain records, its JSON columns decoded.

        node: the claim to read. lane: a fragment of the lane id, every lane when empty.
        every: keep every sample rather than the newest of each cell, which is what a render over
            a repeated-sample program needs and what a representative table must not have.
        """
        store = self.universe.dataset(node)
        frame = store.passing(every=every)
        if not frame.columns:
            return []
        found = frame.to_dicts()
        return [
            store.decoded(row) for row in found if not lane or lane in str(row.get("lane", ""))
        ]


def rendered_twice(figures: Figures, under: Path) -> tuple[str, ...]:
    """Render twice into two directories and name every artifact whose bytes differ.

    An empty answer is the gate passing. A name appearing here is either a file one render wrote
    and the other did not, or the same file with different bytes, and both mean the render carries
    something that is not in the receipts.

    figures: the render to check. under: a scratch directory the two runs land in.
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
