# Content-addressed staging declarations: each yields a CAS key for dedup and an idempotent shell
# command. Nothing here moves a byte; the dispatch preflight (`Dispatcher.submit`'s `verify`
# step) runs the emitted commands.

import hashlib
import shlex
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from patos import FrozenModel

if TYPE_CHECKING:
    from collections.abc import Collection


@runtime_checkable
class Stageable(Protocol):
    """A declared staging need a preflight can check and fetch."""

    @property
    def key(self) -> str:
        """The CAS key identifying this need's content, stable across hosts and runs."""

    def command(self, work_root: str) -> str:
        """An idempotent shell command staging (or checking) this need under `work_root`."""


def _hf_download(
    work_root: str, *, repo_type: str, repo: str, revision: str, include: str = ""
) -> str:
    """One idempotent `hf download` line with `HF_HOME` under `work_root`'s shared cache.

    `work_root` is the shared `/work`-style Lustre root, never job-local scratch, so every trial
    on every host shares one download.

    repo_type: `model` (the `hf download` default, so no flag) or `dataset`.
    include: an `--include` glob, every file when empty.
    """
    flags = ["--repo-type dataset"] if repo_type == "dataset" else []
    if revision:
        flags.append(f"--revision {shlex.quote(revision)}")
    if include:
        flags.append(f"--include {shlex.quote(include)}")
    flag_text = f" {' '.join(flags)}" if flags else ""
    cache = shlex.quote(f"{work_root}/.cache/huggingface")
    return f"HF_HOME={cache} hf download {shlex.quote(repo)}{flag_text}"


class _HfRepo(FrozenModel):
    """A Hugging Face repo (`org/name`) at a pinned revision or branch, `main` when empty."""

    repo: str
    revision: str = ""

    @property
    def key(self) -> str:
        return f"{self.repo}@{self.revision or 'main'}"


class HfModel(_HfRepo):
    """A Hugging Face model repo a study's trials need staged."""

    def command(self, work_root: str) -> str:
        return _hf_download(work_root, repo_type="model", repo=self.repo, revision=self.revision)


class HfDataset(_HfRepo):
    """A Hugging Face dataset repo a study's trials need staged.

    include: an `hf download --include` glob narrowing the pull, every file when empty.
    """

    include: str = ""

    def command(self, work_root: str) -> str:
        return _hf_download(
            work_root,
            repo_type="dataset",
            repo=self.repo,
            revision=self.revision,
            include=self.include,
        )


class RepoFile(FrozenModel):
    """A file checked into this repo a trial reads, `path` relative to the workspace root."""

    path: str

    @property
    def key(self) -> str:
        """The sha256 of the file's current content in the local checkout."""
        return hashlib.sha256(Path(self.path).read_bytes()).hexdigest()

    def command(self, work_root: str) -> str:
        """A preflight check that `path` reached the host via the workspace sync."""
        return f"test -f {shlex.quote(self.path)}"


type Declaration = HfModel | HfDataset | RepoFile


class Needs(tuple[Declaration, ...]):
    """A study's staging declarations: what must be resident before its trials can run."""

    def staging_commands(self, work_root: str) -> list[str]:
        return [item.command(work_root) for item in self]

    def verify(self, host_facts_or_paths: Collection[str]) -> list[Declaration]:
        """Every declared item whose `key` is absent from `host_facts_or_paths`.

        host_facts_or_paths: keys or paths already staged on the target (an `hf` cache listing,
            a probed host's local paths).
        """
        return [item for item in self if item.key not in host_facts_or_paths]
