# The workspace cloned onto the destination, the root at this HEAD and each owned submodule at its
# pointer.

from typing import TYPE_CHECKING

from patos import FrozenModel

from ....core.project import Project
from ....core.section import Section
from ....enums import Verdict

if TYPE_CHECKING:
    from ....fleet.board import Board
    from ....git.repo import Repo
    from ....git.tree import Tree
    from ..carrier import Carrier
    from ..state import Destination


class Cloned(FrozenModel):
    """One repository's line from the destination's clone."""

    repo: str
    outcome: str
    detail: str = ""


class Cloning:
    """The clone of this center's repositories on the destination.

    board: this center's workspace.
    destination: the ssh alias of the machine becoming the center.
    """

    @staticmethod
    def _submodules(tree: Tree) -> list[tuple[Repo, Repo]]:
        """Each checked-out owned submodule with its parent, parents first."""
        return [
            (parent, child)
            for parent in tree.owned()
            for child in parent.children
            if child.initialized and child.owned
        ]

    def __init__(self, board: Board, destination: str) -> None:
        self.board = board
        self.destination = destination

    def run(self, carrier: Carrier, place: Destination) -> list[Section]:
        """The root at this HEAD and each owned submodule at its pointer, then the foreign rest."""
        tree = self.board.git()
        submodules = [
            [parent.name, parent.relative(child), child.declared]
            for parent, child in self._submodules(tree)
        ]
        foreign = [
            child.name
            for parent in tree.owned()
            for child in parent.children
            if child.initialized and not child.owned
        ]
        steps = carrier.call(
            "clone",
            {
                "root": place.root,
                "url": tree.root.url,
                "branch": tree.root.tracking.branch(),
                "commit": tree.root.commits.head(),
                "submodules": submodules,
            },
            list[Cloned],
        )
        again = (
            f"{Project().name} host setup {self.destination} --center --root <an empty directory>"
        )
        rows = [
            Section(
                section=f"clone {step.repo}",
                verdict=Verdict.PASS if step.outcome == "done" else Verdict.FAIL,
                detail=step.detail,
                fix="" if step.outcome == "done" else again,
            )
            for step in steps
        ]
        if foreign:
            rows.append(
                Section(
                    section="clone references",
                    verdict=Verdict.PASS,
                    detail=f"{len(foreign)} foreign submodules left to fetch on demand",
                    fix=f"git submodule update --init -- {foreign[0]}",
                )
            )
        return rows

    def urls(self) -> list[str]:
        """The remote of the root and of each owned submodule, where the clone fetches from."""
        tree = self.board.git()
        return [tree.root.url, *(child.declared for _, child in self._submodules(tree))]
