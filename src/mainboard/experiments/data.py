# Content-addressed staging declarations. A study's trials name what they need (a model, a
# dataset slice, a checked-in file); this module turns those declarations into a CAS key for
# dedup and a shell command a submit preflight can run. It never moves a byte itself: the
# dispatch preflight (`Dispatcher.submit`'s `verify` step, or a future staging step beside it)
# is what actually runs the emitted commands.

import hashlib
import shlex
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from patos import FrozenModel

if TYPE_CHECKING:
    from collections.abc import Collection


@runtime_checkable
class Stageable(Protocol):
    """A declared staging need: something content-addressable a preflight can check and fetch."""

    @property
    def key(self) -> str:
        """The CAS key identifying this need's content, stable across hosts and runs."""

    def command(self, work_root: str) -> str:
        """An idempotent shell command staging (or checking) this need under `work_root`."""


def _hf_download(
    work_root: str, *, repo_type: str, repo: str, revision: str, include: str = ""
) -> str:
    """One idempotent `hf download` line, `HF_HOME` rooted under `work_root`'s shared cache.

    `work_root` is always the `/work`-style shared Lustre root a submit preflight is handed,
    never a job-local scratch directory, so every trial on every host shares the same download
    instead of each re-pulling into its own node-local cache.

    repo_type: `model` (the `hf download` default, so the flag is omitted) or `dataset`.
    revision: a pinned revision/branch, `main` resolved implicitly when empty.
    include: an `hf download --include` glob, every file when empty.
    """
    flags = ["--repo-type dataset"] if repo_type == "dataset" else []
    if revision:
        flags.append(f"--revision {shlex.quote(revision)}")
    if include:
        flags.append(f"--include {shlex.quote(include)}")
    flag_text = f" {' '.join(flags)}" if flags else ""
    cache = shlex.quote(f"{work_root}/.cache/huggingface")
    return f"HF_HOME={cache} hf download {shlex.quote(repo)}{flag_text}"


class HfModel(FrozenModel):
    """A Hugging Face model repo a study's trials need staged.

    repo: the HF repo id (`org/name`).
    revision: a pinned revision or branch, `main` when empty.
    """

    repo: str
    revision: str = ""

    @property
    def key(self) -> str:
        """The CAS key `repo@revision`, an unpinned revision resolving to `main`."""
        return f"{self.repo}@{self.revision or 'main'}"

    def command(self, work_root: str) -> str:
        """An idempotent `hf download` staging this model under `work_root`."""
        return _hf_download(work_root, repo_type="model", repo=self.repo, revision=self.revision)


class HfDataset(FrozenModel):
    """A Hugging Face dataset repo a study's trials need staged.

    repo: the HF dataset repo id (`org/name`).
    include: an `hf download --include` glob narrowing the pull, every file when empty.
    revision: a pinned revision or branch, `main` when empty.
    """

    repo: str
    include: str = ""
    revision: str = ""

    @property
    def key(self) -> str:
        """The CAS key `repo@revision`, an unpinned revision resolving to `main`."""
        return f"{self.repo}@{self.revision or 'main'}"

    def command(self, work_root: str) -> str:
        """An idempotent `hf download` staging this dataset under `work_root`."""
        return _hf_download(
            work_root,
            repo_type="dataset",
            repo=self.repo,
            revision=self.revision,
            include=self.include,
        )


class RepoFile(FrozenModel):
    """A file already checked into this repo a trial reads (a tokenizer, a small fixture).

    path: the file's path, relative to the workspace root.
    """

    path: str

    @property
    def key(self) -> str:
        """The CAS key: the sha256 of the file's current content, read from the local checkout."""
        return hashlib.sha256(Path(self.path).read_bytes()).hexdigest()

    def command(self, work_root: str) -> str:
        """A preflight check confirming `path` reached the host via the workspace sync."""
        return f"test -f {shlex.quote(self.path)}"


type Declaration = HfModel | HfDataset | RepoFile


class Needs(tuple[Declaration, ...]):
    """A study's staging declarations: what must be resident before its trials can run."""

    def staging_commands(self, work_root: str) -> list[str]:
        """Every declared item's staging command, suitable for a submit preflight to run."""
        return [item.command(work_root) for item in self]

    def verify(self, host_facts_or_paths: Collection[str]) -> list[Declaration]:
        """Every declared item whose `key` is absent from `host_facts_or_paths`.

        host_facts_or_paths: keys or paths already known staged on the target (an `hf` cache
            listing, a probed host's local paths).
        """
        return [item for item in self if item.key not in host_facts_or_paths]
