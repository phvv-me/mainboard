# Everything a consumer states, in one object, so its conftest is a declaration and not a program:
# where its trials live, which words it settles on, which process-global flags its lanes may move,
# and which working tree stamps the commit. The plugin is the rest.

from collections.abc import Callable, Mapping
from pathlib import Path

from patos import FrozenModel, Runtime
from pydantic import Field

from ..profile.profiler import Collection
from .artifacts import Artifact
from .flags import Flag
from .universe import Universe
from .vocabulary import Vocabulary

# The default marker table. `gpu`, `slow` and `paid` decide whether a trial runs; only `paid` is
# wired to an option, since its cost is money. `adversarial` and `search` name an adaptive lane
# kind, whose result is a candidate rather than coverage: `-m "not adversarial and not search"`
# is a run that claims only what a declared grid measured.
MARKERS = {
    "gpu": "needs a real card, skipped where there is none",
    "slow": "runs for minutes rather than seconds",
    "paid": "could bill money, skipped unless --paid is passed",
    "adversarial": "hunts a counterexample by shrinking, so what it finds is a candidate",
    "search": "proposes its own points adaptively, so what it finds is a candidate",
    "phase": "the registered phase its receipts belong to, a coverage axis when one is declared",
}


class Declaration(FrozenModel):
    """One workspace's trials, stated once and read by every hook the plugin implements.

    universe: where the trials live and what scopes their coverage.
    words: the settle words this workspace uses, whose meaning is its own.
    flags: the process-global values a lane may move, recorded on every receipt and refused at
        the end of a run if any is left off its baseline.
    repo: the local source root to capture, the universe root when unset. A dispatched
        closure names its own workspace root independently of this nested project.
    resident: reads the bytes a claim's holdings currently occupy, so leaving a claim can be
        checked rather than assumed. Unset skips the check and the holdings still drop on time.
    """

    universe: Universe
    words: Vocabulary = Field(default_factory=Vocabulary)
    flags: tuple[Runtime[Flag], ...] = ()
    repo: Path | None = None
    markers: Mapping[str, str] = Field(default_factory=lambda: MARKERS.copy())
    resident: Callable[[], int] | None = None
    inputs: Mapping[str, Artifact] = Field(default_factory=dict)
    collection: Collection = Field(default_factory=Collection)

    @property
    def tree(self) -> Path:
        """The local source root to capture when no dispatched closure is supplied."""
        return self.repo or self.universe.root
