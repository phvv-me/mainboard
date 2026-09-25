# The whole matrix before a push: the gate here, and at the same time on one declared host of each
# other platform the package supports, with one table at the end. The hosts come from the
# workspace's `[ci] hosts`, and a host of this machine's own family is passed over because the
# local leg already speaks for it, which is what lets one list serve whichever machine is the
# center today. A platform no leg covers is named rather than silently counted as passing.

from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

from ..core.errors import MissionError
from ..dispatch.mirror import Mirror
from ..dispatch.sync import patterns
from .legs import Leg, LocalLeg, RemoteLeg

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..board import Board
    from ..context.plan import ExecutionPlan
    from ..dispatch.dispatcher import Dispatcher
    from .definition import Family, Package
    from .legs import Result


class PackageMirror:
    """Ships one package of the working tree, as it stands, under a root of its own on a host.

    The workspace's own mirror agent does the sending, over the one ssh channel on every
    platform, with the host's excludes kept and the scope narrowed to the package. Nothing there
    is protected, since that root holds only this package and what its gate built from it, which
    git ignores and a mirror therefore never prunes.
    """

    def __init__(self, dispatcher: Dispatcher) -> None:
        self.dispatcher = dispatcher

    def __call__(self, plan: ExecutionPlan, root: str, package: str) -> None:
        """Send `package` to `root` on `plan`'s host."""
        mirror = Mirror(self.dispatcher.root, self.dispatcher.agent(plan))
        mirror.push(root, scopes=[self.dispatcher.scope(plan, [package])], protected=patterns([]))


class Matrix:
    """A package's gate on every leg at once.

    package: the package and its gate.
    legs: the machines it runs on, this one among them when it is a supported platform.
    """

    def __init__(self, package: Package, legs: Sequence[Leg]) -> None:
        self.package = package
        self.legs = legs

    @classmethod
    def planned(cls, package: Package, board: Board) -> Matrix:
        """This machine plus every `[ci]` host of `board` of a supported family this one is not."""
        if not package.root.is_relative_to(board.root):
            raise MissionError(f"{package.root} is outside the workspace at {board.root}")
        relative = package.root.relative_to(board.root).as_posix()
        local = LocalLeg(package.root)
        ship = PackageMirror(board.dispatcher)
        remote = [
            RemoteLeg(board.on(host).plan(container="none"), relative, ship=ship)
            for host in board.manifest.ci.hosts
        ]
        others = [leg for leg in remote if leg.family != local.family]
        supported = package.definition.os
        return cls(package, [leg for leg in [local, *others] if leg.family in supported])

    @property
    def uncovered(self) -> list[Family]:
        """The supported families no leg runs on."""
        covered = {leg.family for leg in self.legs}
        return [family for family in self.package.definition.os if family not in covered]

    def run(self) -> list[Result]:
        """Every leg's results, the legs run at once and reported in their declared order."""
        gate = self.package.definition
        with ThreadPoolExecutor(max_workers=len(self.legs)) as pool:
            settled = list(pool.map(lambda leg: list(leg.run(gate.on(leg.family))), self.legs))
        return [result for results in settled for result in results]
