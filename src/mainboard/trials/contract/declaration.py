# Everything a consumer states, in one object, so its conftest is a declaration and not a program:
# where its trials live, which words it settles on, which process-global flags its lanes may move,
# and which working tree stamps the commit. The plugin is the rest.

from collections.abc import Callable, Mapping
from pathlib import Path

from patos import FrozenModel
from pydantic import Field

from ...profile.profiler import Collection
from ..artifacts import Artifact
from ..store.constants import MARKERS
from ..store.universe import Universe
from .flags import Flag
from .vocabulary import Vocabulary


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
    flags: tuple[Flag, ...] = ()
    repo: Path | None = None
    markers: Mapping[str, str] = Field(default_factory=lambda: MARKERS.copy())
    resident: Callable[[], int] | None = None
    inputs: Mapping[str, Artifact] = Field(default_factory=dict)
    collection: Collection = Field(default_factory=Collection)

    @property
    def tree(self) -> Path:
        """The local source root to capture when no dispatched closure is supplied."""
        return self.repo or self.universe.root
