# The whole matrix before a push: the gate here, and at the same time on one declared host of each
# other platform the package supports, with one table at the end. The hosts come from the
# workspace's `[ci] hosts`, and a host of this machine's own family is passed over because the
# local leg already speaks for it, which is what lets one list serve whichever machine is the
# center today. A platform no leg covers is named rather than silently counted as passing.

from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

from ..core.errors import MissionError
from ..manifest.schema.host import Sync
from .legs import Leg, LocalLeg, RemoteLeg

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..board import Board
    from ..context.plan import ExecutionPlan
    from ..dispatch.dispatcher import Dispatcher
    from .definition import Family, Package
    from .legs import Result


class Mirror:
    """Ships one package of the working tree, as it stands, under a root of its own on a host.

    The workspace's own mirror machinery does the sending, rsync on POSIX and tar on Windows,
    with the host's excludes and protections kept and its include list narrowed to the package.

    dispatcher: the workspace's dispatch core.
    """

    def __init__(self, dispatcher: Dispatcher) -> None:
        self.dispatcher = dispatcher

    def __call__(self, plan: ExecutionPlan, root: str, package: str) -> None:
        """Send `package` to `root` on `plan`'s host."""
        scope = plan.profile.sync
        narrowed = Sync(include=[package], exclude=scope.exclude, protect=scope.protect)
        profile = plan.profile.model_copy(update={"sync": narrowed})
        self.dispatcher.rsync_up(plan.model_copy(update={"profile": profile}), root)


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
        """This machine plus every `[ci]` host of a supported family this machine is not.

        package: the package whose gate runs.
        board: the workspace the package lives in, which declares the hosts.
        """
        if not package.root.is_relative_to(board.root):
            raise MissionError(f"{package.root} is outside the workspace at {board.root}")
        relative = package.root.relative_to(board.root).as_posix()
        local = LocalLeg(package.root)
        ship = Mirror(board.dispatcher)
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
        with ThreadPoolExecutor(max_workers=len(self.legs)) as pool:
            settled = list(pool.map(self._leg, self.legs))
        return [result for results in settled for result in results]

    def _leg(self, leg: Leg) -> list[Result]:
        return list(leg.run(self.package.definition.on(leg.family)))
