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
# declares no search should not install a sampler. `driver` is the one import seam, and a missing
# package refuses by naming both the package and the extra that carries it rather than surfacing a
# bare `ModuleNotFoundError` from three frames down.

from importlib import import_module
from types import ModuleType

from patos import FrozenModel


class Absent(ImportError):
    """An adaptive lane whose driver package this environment does not carry."""


def driver(package: str, extra: str) -> ModuleType:
    """One adaptive lane's driver, refusing by naming the package and the extra that ships it.

    package: the importable name. extra: the `mainboard[...]` extra that installs it.
    """
    try:
        return import_module(package)
    except ImportError as missing:
        raise Absent(
            f"this lane is driven by {package!r}, which is an optional extra of this tool and is "
            f"not installed here. Install it with `pip install mainboard[{extra}]`, or declare "
            f"{package!r} in the workspace manifest that owns the lane."
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
