# Content-addressed staging declarations: each yields a CAS key for dedup. Nothing here moves a
# byte; a gate checks which keys a host lacks.

import hashlib
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

from patos import FrozenModel

if TYPE_CHECKING:
    from collections.abc import Collection


class Stageable(ABC, FrozenModel):
    """A declared staging need a preflight can check."""

    @property
    @abstractmethod
    def key(self) -> str:
        """The CAS key identifying this need's content, stable across hosts and runs."""


class _HfRepo(Stageable):
    """A Hugging Face repo (`org/name`) at a pinned revision or branch, `main` when empty."""

    repo: str
    revision: str = ""

    @property
    def key(self) -> str:
        return f"{self.repo}@{self.revision or 'main'}"


class HfModel(_HfRepo):
    """A Hugging Face model repo a study's trials need staged."""


class HfDataset(_HfRepo):
    """A Hugging Face dataset repo a study's trials need staged."""


class RepoFile(Stageable):
    """A file checked into this repo a trial reads, `path` relative to the workspace root."""

    path: str

    @property
    def key(self) -> str:
        """The sha256 of the file's current content in the local checkout."""
        return hashlib.sha256(Path(self.path).read_bytes()).hexdigest()


type Declaration = HfModel | HfDataset | RepoFile


class Needs(tuple[Declaration, ...]):
    """A study's staging declarations: what must be resident before its trials can run."""

    def verify(self, host_facts_or_paths: Collection[str]) -> list[Declaration]:
        """Every declared item whose `key` is absent from `host_facts_or_paths`.

        host_facts_or_paths: keys or paths already staged on the target (an `hf` cache listing,
            a probed host's local paths).
        """
        return [item for item in self if item.key not in host_facts_or_paths]
