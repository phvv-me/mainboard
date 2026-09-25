# The immutable copy of the synced source a dispatch pins its job to, so a later mirror sync can
# never rewrite the code a queued or running job imports.
#
# The mirror stays the transfer's target, since incremental transfer is what makes dispatch cheap.
# A job runs from a snapshot of it taken at submit time and named for the source identity its
# receipts carry, so two dispatches of one tree share a snapshot. The compiled environment, the
# dispatch state and the results path are symlinks back to the mirror, so a job activates the
# mirror's environment, logs where every read looks, and leaves results where the pull goes.
#
# Two images exist. A command that ships the mirror pins the whole synced allowlist, each directory
# the copy created filled with links back to whatever else the mirror holds there. A job spelled by
# file pins its closure alone: the mirror is reachable only through the environment, the data
# paths the job declared, and its results path.
#
# The generated tree is where a symlink is not enough: its manifest carries the root it was
# compiled for and the tool resolves it through the directory it stands in, so a symlinked tree
# would send every task back to the mirror and pin nothing. Its directories are real, its files
# hardlinked, and only the installed artifacts under them symlinked, pinning the tree while
# leaving one environment per host.
#
# The pin runs in the target's standard-library agent under its own kernel file lock, so it needs
# no POSIX shell or tool beyond Python. Everything a caller typed is refused here first.

from abc import ABC, abstractmethod
from pathlib import PurePosixPath
from typing import TYPE_CHECKING

from patos import FrozenModel

from ..core.project import Project
from ..jobs.pins import STAGING as PINS
from .agent import AgentRefused
from .agent.program import CLOSURE, STAMP, WRAPPERS, ScopeSpec
from .shared import state_dir

if TYPE_CHECKING:
    from .agent import Agent
    from .agent.program import ImageSpec

# Under the state directory every transfer excludes, so a sync neither ships nor prunes one.
SOURCES = f"{state_dir()}/sources"


def stamped(key: str, *, commit: str, digest: str) -> str:
    """A finished snapshot's stamp: the key alone on the first line, as every existing stamp holds
    it, then `commit`/`digest` as `name value` lines (each omitted when empty), which is how a job
    on a machine without history says what it runs.
    """
    lines = [
        key,
        *(f"{name} {value}" for name, value in (("commit", commit), ("digest", digest)) if value),
    ]
    return "\n".join(lines) + "\n"


def writable(path: str) -> str:
    """`path` as a workspace-relative results path, empty when absolute or climbing out.

    It is the one caller-typed string a snapshot turns into a link back at the mirror, so it is
    refused here rather than handed to the pin that removes what stands there.
    """
    posix = PurePosixPath(path)
    if not path or posix.is_absolute() or ".." in posix.parts:
        return ""
    return posix.as_posix()


def _hex_sha256(text: str) -> bool:
    return len(text) == 64 and all(char in "0123456789abcdef" for char in text)


def _reserved(path: str) -> bool:
    """Whether `path` would replace one of a snapshot's own control files."""
    return path in (".", CLOSURE, STAMP, WRAPPERS) or path.startswith(WRAPPERS + "/")


class Image(ABC, FrozenModel):
    """What one snapshot copies out of the mirror, and what it reaches back into the mirror for."""

    @abstractmethod
    def request(self, digest: str, results: str) -> ImageSpec:
        """This image as the pin's request, refusing what the host must not see.

        results: the dispatch's declared results path, empty for none.
        """


class Mirrored(Image):
    """The whole synced scope, what a command that ships the mirror runs from.

    scope: the roots and rules the mirror shipped by, so the snapshot holds the shipped file set
        and not the artifacts the host wrote beside it.
    """

    scope: ScopeSpec

    def request(self, digest: str, results: str) -> ImageSpec:
        """The scope, whatever the digest: a command's tree carries no listing to check."""
        del digest, results
        return {"kind": "mirrored", "scope": self.scope}


class Sealed(Image):
    """A job's closure and nothing beside it, what a job spelled by file runs from.

    listing: the mirror's workspace-relative closure listing, first column naming every shipped
        file. Frozen as `CLOSURE`, which the runner reads through `MAINBOARD_CLOSURE` rather than
        a later mirror's listing.
    needs: data paths the job reads, linked back to the mirror on every dispatch. A need the
        mirror lacks refuses the dispatch by name, since a dangling link fails after the queue.
    pins: staged Hub pins, checked on the mirror and reached through their shared staging
        directory.
    """

    listing: str
    needs: tuple[str, ...] = ()
    pins: tuple[str, ...] = ()

    def request(self, digest: str, results: str) -> ImageSpec:
        """The listing and its live paths, refusing a digest or live path the pin cannot trust."""
        if not _hex_sha256(digest):
            raise ValueError("a sealed snapshot requires its complete closure digest")
        live = [writable(path) for path in (*self.needs, *([results] if results else []))]
        if any(not path or _reserved(path) for path in live):
            raise ValueError("snapshot data and results must be relative paths below the root")
        return {
            "kind": "sealed",
            "listing": self.listing,
            "digest": digest,
            "needs": live[: len(self.needs)],
            "pins": list(self.pins),
            "staging": PINS,
            "live": live,
        }


class Snapshots:
    """The pinned source trees on one host, under `{root}/{STATE_DIR}/sources/`.

    Source files share inodes with the mirror; directories and frozen metadata add storage.
    Hardlinking also makes the pin correct: the mirror replaces a changed file by renaming a new
    one over it, so only the mirror's entry moves to the new inode. Automatic deletion is unsafe,
    since one workstation's job cache cannot establish ownership across other dispatchers or the
    gap before a submitted job is recorded: snapshots remain until an operator verifies no queued
    or running job uses them. Watch inode quotas.

    root: the host's workspace root, the mirror every snapshot is taken from and links back to.
    """

    def __init__(self, root: str) -> None:
        self.root = root.rstrip("/")

    @property
    def base(self) -> str:
        return f"{self.root}/{SOURCES}"

    def path(self, key: str) -> str:
        """Where `key` is pinned, materialised or not: pure arithmetic, because a dispatch renders
        the job script that runs from it before opening the connection that creates it."""
        return f"{self.base}/{key}"

    @staticmethod
    def script(staged: str) -> str:
        """The frozen path of a generated wrapper, whose filename carries its complete digest."""
        name = PurePosixPath(staged).name
        digest = name.removeprefix("job-").removesuffix(".sh")
        if (
            not staged
            or writable(staged) != staged
            or name != f"job-{digest}.sh"
            or not _hex_sha256(digest)
        ):
            raise ValueError("a staged wrapper requires a canonical path and full SHA-256 name")
        return f"{WRAPPERS}/{name}"

    def pin(
        self,
        agent: Agent,
        *,
        key: str,
        image: Image,
        results: str = "",
        prefix: str = "",
        environment: str = "default",
        commit: str = "",
        digest: str = "",
        script: str = "",
    ) -> str:
        """Materialise the snapshot for `key` on the host and answer the path a job runs from.

        A key already pinned is verified without rebuilding. Its results path, needs and wrapper
        are linked on every dispatch all the same, since they belong to the dispatch and two
        batches off one commit routinely declare different ones.

        agent: the host's agent, which builds the tree under the host's pin lock.
        results: linked back to the mirror so what the job writes there is what a later pull
            brings home; empty for none.
        prefix: the immutable, content-named environment this tree activates. Written once when
            the tree is built and never repointed, so a queued wave keeps it however often the
            workspace re-solves. Empty reaches the mirror's own environment, as a workspace with
            no addressed prefixes does.
        environment: the environment `prefix` belongs to, the generated-tree directory the link
            is written into.
        script: staged generated wrapper to freeze and verify before submission; empty for a
            direct remote command whose file is not shipped.
        commit / digest: the tree's provenance for the stamp, the only place on a history-less
            mirror either can be read.
        """
        if not key or key in (".", "..") or PurePosixPath(key).name != key:
            raise ValueError("a snapshot key must be one directory name")
        path = self.path(key)
        described = image.request(digest, results)
        relative = writable(results)
        if relative and _reserved(relative):
            raise ValueError("a results path must not replace snapshot control files")
        wrapper = self.script(script) if script else ""
        try:
            agent.ask(
                {
                    "pin": {
                        "root": self.root,
                        "base": self.base,
                        "key": key,
                        "stamp": stamped(key, commit=commit, digest=digest),
                        "out": Project().out_dir,
                        "prefix": prefix,
                        "environment": environment,
                        "results": relative,
                        "script": script,
                        "wrapper": wrapper,
                        "image": described,
                    }
                }
            )
        except AgentRefused as refused:
            raise SystemExit(f"could not pin the source tree at {path}: {refused}") from None
        return path
