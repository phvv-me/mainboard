# EVERYTHING A CONSUMER STATES, IN ONE OBJECT, SO ITS CONFTEST IS A DECLARATION AND NOT A PROGRAM.
#
# The reference this generalizes carried the settle words, the marker table, the coverage rule,
# the provenance probe and the arithmetic pin in one hand-written conftest per project, and every
# project that copied it copied the defects too. Here a project states four things, where its
# trials live, what words it settles on, which process-global flags its lanes are allowed to move,
# and which working tree stamps the commit, and the plugin is the rest.

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .universe import Universe
from .vocabulary import Vocabulary

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from pathlib import Path

    from .flags import Flag

# What a trial can be marked with out of the box, and the two things a marker here does. The first
# three decide whether a trial RUNS AT ALL, and `paid` is the only one wired to an option since it
# is the only one whose cost is money rather than time. The last two decide nothing and NAME THE
# LANE KIND, because an adaptive lane's result is a candidate rather than coverage and a reader
# tallying what a suite establishes has to be able to select those out: `-m "not adversarial and
# not search"` is the whole of a run that claims only what a declared grid measured.
MARKERS = {
    "gpu": "needs a real card, skipped where there is none",
    "slow": "runs for minutes rather than seconds",
    "paid": "could bill money, skipped unless --paid is passed",
    "adversarial": "hunts a counterexample by shrinking, so what it finds is a candidate",
    "search": "proposes its own points adaptively, so what it finds is a candidate",
}


@dataclass(frozen=True, slots=True)
class Declaration:
    """One workspace's trials, stated once and read by every hook the plugin implements.

    universe: where the trials live and what scopes their coverage.
    words: the settle words this workspace uses, whose names are its own and whose meaning is
        nobody else's business.
    flags: the process-global values a lane may move, recorded on every receipt and refused at
        the end of a run if any is left off its baseline.
    repo: the working tree whose commit stamps every receipt, the universe root when unset.
    markers: the marker table registered for this session.
    resident: reads the bytes a claim's holdings currently occupy, so leaving a claim can be
        checked rather than assumed. Unset skips the check and the holdings still drop on time.
    """

    universe: Universe
    words: Vocabulary = field(default_factory=Vocabulary)
    flags: tuple[Flag, ...] = ()
    repo: Path | None = None
    markers: Mapping[str, str] = field(default_factory=lambda: MARKERS)
    resident: Callable[[], int] | None = None

    @property
    def tree(self) -> Path:
        """The working tree a receipt's commit is probed from."""
        return self.repo or self.universe.root
