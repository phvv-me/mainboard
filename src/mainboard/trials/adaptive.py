# Adaptive lanes, and the one rule that binds both kinds.
#
# A declared lane states its grid before it runs; an adaptive lane picks its next operand or shape
# from what the previous ones scored, so its grid exists only after the search and two runs visit
# different points. An adaptive result is therefore a CANDIDATE, never coverage: a shrunk witness
# or a sampled worst point says where to look, not how often the thing happens. Whatever a claim
# leans on is confirmed by a declared parametrize cell on fresh seeds first (search proposes, the
# grid confirms), and `Owed` writes that debt onto the receipt.
#
# The drivers are optional extras, imported through `driver`, which is keyed by the lane kind so
# the kind, its marker and its extra are one word. The plugin warms each collected kind's driver
# at collection; `pytest_plugin._warmed` says why.

from importlib import import_module
from types import ModuleType

from patos import FrozenModel

# Which package drives each lane kind; the key is also the marker and the `mainboard[...]` extra.
DRIVERS = {"adversarial": "hypothesis", "search": "optuna"}


class Absent(ImportError):
    """An adaptive lane whose driver package this environment does not carry."""


def driver(kind: str) -> ModuleType:
    """One lane kind's driver, refusing by naming the package and the extra that ships it."""
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
    cell: the parametrize coordinate inside that lane, making the confirmation a grid point.
    seeds: what the confirmation must run on.
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
