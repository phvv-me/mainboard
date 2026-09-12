"""Hub pins a job declares as needs: `hf://<repo>@<revision>/<filename>`, shipped from the cache.

A tokenizer or a config a job reads at a pinned revision is data the job needs on the host as
much as a corpus is, and a job that opens it offline fails on a host whose cache never held it.
So a need may name the Hub entry itself; the dispatch stages the locally cached file under the
workspace at `.mainboard/pins` in the cache's own layout, ships it like any other need, and the
runner points the Hub client at that directory, so an offline read finds exactly what was pinned.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

from patos import FrozenModel

from ..core.errors import MissionError

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

PREFIX = "hf://"
# Where staged pins sit under the workspace, in the Hub cache layout; a job's Hub client is
# pointed here by the runner.
STAGING = ".mainboard/pins"


class Pin(FrozenModel):
    """One Hub file at one revision.

    repo: the repository id, `org/name`.
    revision: the commit the file is pinned at.
    filename: the file inside the repository.
    """

    repo: str
    revision: str
    filename: str

    @classmethod
    def parse(cls, spec: str) -> Pin:
        """A pin from its spelling, `hf://org/name@revision/filename`."""
        body = spec.removeprefix(PREFIX)
        repo, at, rest = body.partition("@")
        revision, slash, filename = rest.partition("/")
        if not (spec.startswith(PREFIX) and at and slash and repo.count("/") == 1 and filename):
            raise MissionError(f"a pin is spelled hf://org/name@revision/filename, not {spec!r}")
        return cls(repo=repo, revision=revision, filename=filename)

    @property
    def relative(self) -> str:
        """The file's path inside a Hub cache, the layout the Hub client reads offline."""
        folder = "models--" + self.repo.replace("/", "--")
        return f"{folder}/snapshots/{self.revision}/{self.filename}"

    def cached(self, cache: Path | None = None) -> Path:
        """Where this machine's Hub cache holds the file, whether or not it is there."""
        return (cache or hub_cache()) / self.relative


def hub_cache() -> Path:
    """This machine's Hub cache directory, as the Hub client would resolve it."""
    if explicit := os.environ.get("HF_HUB_CACHE"):
        return Path(explicit)
    return Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub"


def is_pin(need: str) -> bool:
    """Whether a declared need names a Hub pin rather than a workspace path."""
    return need.startswith(PREFIX)


def split(needs: Iterable[str]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """The declared needs as workspace paths and Hub pins, each in declaration order."""
    listed = list(needs)
    return (
        tuple(need for need in listed if not is_pin(need)),
        tuple(need for need in listed if is_pin(need)),
    )


def stage(specs: Sequence[str], root: Path, cache: Path | None = None) -> tuple[str, ...]:
    """Copy every pin from this machine's Hub cache under the workspace, answering their paths.

    A pin the cache does not hold refuses the dispatch by name, with the command that fetches it.

    specs: the pins as declared.
    root: the workspace root.
    cache: the Hub cache to read, this machine's when omitted.
    """
    staged = []
    for spec in specs:
        pin = Pin.parse(spec)
        source = pin.cached(cache)
        if not source.is_file():
            raise MissionError(
                f"the pin {spec} is not in this machine's Hub cache at {source}; fetch it with "
                f"`hf download {pin.repo} {pin.filename} --revision {pin.revision}` first"
            )
        relative = PurePosixPath(STAGING) / pin.relative
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.is_file() or target.stat().st_size != source.stat().st_size:
            shutil.copyfile(source, target)
        staged.append(relative.as_posix())
    return tuple(staged)
