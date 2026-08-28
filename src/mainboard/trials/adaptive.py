# ADAPTIVE LANES, AND THE ONE RULE THAT BINDS BOTH KINDS OF THEM.
#
# A declared lane states its grid before it runs, so being collected IS its declaration and the
# completeness rule can say what the store still owes. An ADAPTIVE lane cannot do that. It chooses
# its next operand, or its next shape, from what the previous ones scored, so its grid does not
# exist until the search is over and two runs of it visit two different sets of points.
#
# SO AN ADAPTIVE RESULT IS A CANDIDATE AND NEVER COVERAGE. A witness a shrinker minimised and a
# worst point a sampler walked to are both proposals: they say WHERE to look, and nothing about
# how often the thing happens, because the place they name is the place the search was steered
# toward. Whatever a claim ends up leaning on is confirmed by a DECLARED PARAMETRIZE CELL ON FRESH
# SEEDS before it leans on it, which is the house successor rule with the search in front of it:
# search proposes and the grid confirms. `Owed` is that debt written onto the receipt, so a reader
# who finds a candidate also finds the cell that owes its confirmation and can see whether it was
# ever paid.
#
# AND THE DRIVERS ARE OPTIONAL EXTRAS. Neither lane kind's package is a dependency of this tool,
# because a workspace that declares no adversarial lane should not install a shrinker and one that
# declares no search should not install a sampler. `driver` is the one import seam, keyed by the
# LANE KIND rather than by the package, so the kind, its marker and its extra are one name spelled
# in one place; a missing package refuses by naming both rather than surfacing a bare
# `ModuleNotFoundError` from three frames down.
#
# AND A DRIVER IS WARMED AT COLLECTION, WHICH IS NOT A PERFORMANCE CHOICE. A package imported for
# the first time inside a running test leaves that test's frame reachable, the frame holds the
# fixture values pytest passed it, and those are a claim's checkpoint, so `Stage` reports a card
# that never came back and refuses the run. It cost 1.33 GB and an afternoon to find. The plugin
# therefore imports each collected kind's driver before any trial runs, which also turns a missing
# package into a refusal at collection instead of one in the middle of a measurement.

from importlib import import_module
from types import ModuleType

from patos import FrozenModel

# Which package drives each adaptive lane kind. The key is the kind, which is also its marker and
# also the `mainboard[...]` extra that installs it, so a consumer learns one word.
DRIVERS = {"adversarial": "hypothesis", "search": "optuna"}


class Absent(ImportError):
    """An adaptive lane whose driver package this environment does not carry."""


def driver(kind: str) -> ModuleType:
    """One adaptive lane kind's driver, refusing by naming the package and the extra that ships it.

    kind: the lane kind, which is the marker a lane carries and the extra that installs its driver.
    """
    package = DRIVERS[kind]
    try:
        return import_module(package)
    except ImportError as missing:
        raise Absent(
            f"a {kind!r} lane is driven by {package!r}, which is an optional extra of this tool "
            f"and is not installed here. Install it with `pip install mainboard[{kind}]`, or "
            f"declare {package!r} in the workspace manifest that owns the lane."
        ) from missing


class Owed(FrozenModel):
    """The declared cell that owes an adaptive candidate its confirmation, on fresh seeds.

    lane: the declared lane the confirmation runs in, spelled as a caller would select it.
    cell: the parametrize coordinate inside that lane, which is what makes the confirmation a
        grid point rather than a second search.
    seeds: what the confirmation must run on, `fresh` unless a claim has a reason to say more.
    """

    lane: str
    cell: dict[str, str] = {}
    seeds: str = "fresh"

    @property
    def stated(self) -> str:
        """This debt as the one sentence a receipt's reason carries."""
        where = ", ".join(f"{name}={value}" for name, value in sorted(self.cell.items()))
        return (
            f"a CANDIDATE, owed confirmation by the declared cell {self.lane}"
            f"{f'[{where}]' if where else ''} on {self.seeds} seeds"
        )
