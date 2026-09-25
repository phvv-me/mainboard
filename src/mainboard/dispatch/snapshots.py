# The immutable copy of the synced source a dispatch pins its job to, so a later mirror sync
# can never rewrite the code a job that is already queued or running imports.
#
# The mirror stays the transfer's target, because an incremental transfer is what makes
# dispatching cheap at all. What a job runs from is a snapshot of that mirror taken at submit
# time and named for the source identity its own receipts carry, so two dispatches of one tree
# share a snapshot and a dispatch of a different tree gets its own. A snapshot holds what its
# image says and nothing else: the compiled environment, the dispatch state and the dispatch's
# own results path are symlinks back to the mirror, so a job in a snapshot activates the
# mirror's environment, writes its log where every later read already looks, and leaves its
# results where the pull already goes.
#
# Two images exist. A command that ships the mirror pins the whole synced allowlist, and every
# directory that copy created is filled with links back to whatever else the mirror holds there.
# A job spelled by file pins its closure, the exact files it imports and nothing beside them: the
# mirror is not reachable from that tree except through the environment, the data paths the job
# declared it needs, and its results path.
#
# The generated tree is the one place those two worlds meet, and it is the one place a symlink
# is not enough. A workspace's generated manifest carries the root it was compiled for, and the
# tool resolves that manifest through the directory it is standing in, so a snapshot whose
# generated tree were a symlink would send every task back to the mirror it points at and pin
# nothing. Its directories are therefore real here, its files hardlinked, and only the installed
# artifacts under them symlinked, which pins the tree while leaving one environment for the host.
#
# The pin itself runs in the target's agent, the same standard-library program the mirror talks
# to, under a kernel file lock of its own, so pinning needs neither a POSIX shell nor any tool
# on the host beyond Python. Everything a caller typed is refused here, before the host is asked.

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

# Where a host keeps its pinned trees, under the dispatch state directory the mirror already
# excludes from every transfer, so a sync can neither ship one nor prune one.
SOURCES = f"{state_dir()}/sources"


def stamped(key: str, *, commit: str, digest: str) -> str:
    """What a finished snapshot's stamp holds: the tree's key, and the provenance of that tree.

    The key stays the first line and stays alone on it, because it is what the tree is named for
    and what every earlier stamp on every host already holds. The provenance follows as `name
    value` lines, which is what a job reads to say which commit it is running and what the
    shipped bytes hashed to, on a machine that has no history to derive either from.

    key: the tree's identity, the source's key.
    commit / digest: the dispatching tree's commit and content digest, either empty when the
        workspace has no git to answer with.
    """
    lines = [
        key,
        *(f"{name} {value}" for name, value in (("commit", commit), ("digest", digest)) if value),
    ]
    return "\n".join(lines) + "\n"


def writable(path: str) -> str:
    """`path` as a workspace-relative results path, empty when it is not one.

    A results path is the one caller-typed string a snapshot turns into a link pointing back at
    the mirror, so an absolute path or one climbing out of the workspace is refused here rather
    than handed to the pin that removes what stands there.
    """
    posix = PurePosixPath(path)
    if not path or posix.is_absolute() or ".." in posix.parts:
        return ""
    return posix.as_posix()


def _reserved(path: str) -> bool:
    """Whether `path` would replace one of a snapshot's own control files."""
    return path in (".", CLOSURE, STAMP, WRAPPERS) or path.startswith(WRAPPERS + "/")


class Image(ABC, FrozenModel):
    """What one snapshot copies out of the mirror, and what it reaches back into the mirror for."""

    @abstractmethod
    def request(self, digest: str, results: str) -> ImageSpec:
        """This image as the pin's request describes it, refusing what the host must not see.

        digest: the dispatching tree's content digest.
        results: the dispatch's declared results path, empty for none.
        """


class Mirrored(Image):
    """The whole synced scope, what a command that ships the mirror runs from.

    scope: the roots the mirror shipped and the rules it shipped them by, so the snapshot holds
        the shipped file set and not the artifacts the host wrote beside it. Every directory the
        copy creates is then filled with links back to whatever else the mirror holds there.
    """

    scope: ScopeSpec

    def request(self, digest: str, results: str) -> ImageSpec:
        """The scope, whatever the digest: a command's tree carries no listing to check."""
        del digest, results
        return {"kind": "mirrored", "scope": self.scope}


class Sealed(Image):
    """A job's closure and nothing beside it, what a job spelled by file runs from.

    listing: the closure listing the mirror carries, workspace-relative, whose first column
        names every shipped file. The snapshot freezes it as `CLOSURE`; the runner reads that
        copy through `MAINBOARD_CLOSURE`, not a later mirror's listing.
    needs: the workspace-relative data paths the job reads, each linked back to the mirror on
        every dispatch. A need the mirror does not hold refuses the dispatch by name, since a
        job that opens a dangling link fails after the queue rather than before it.
    pins: staged Hub pins, each checked on the mirror and reached through the one staging
        directory they share.
    """

    listing: str
    needs: tuple[str, ...] = ()
    pins: tuple[str, ...] = ()

    def request(self, digest: str, results: str) -> ImageSpec:
        """The listing and its live paths, refusing a digest or live path the pin cannot trust."""
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
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
    Automatic deletion is unsafe: one workstation's job cache cannot establish ownership
    across other dispatchers or the gap before a submitted job is recorded. Snapshots remain
    until an operator verifies that no queued or running job uses them. Watch inode quotas.

    Hardlinking is also what makes the pin correct rather than merely cheap. The mirror replaces
    a changed file by writing a new one and renaming it over the old name, so the mirror's
    directory entry moves to a new inode while the snapshot's entry keeps pointing at the one
    the job is already reading.

    root: the workspace root on the host, the mirror every snapshot is taken from and links
        back to.
    """

    def __init__(self, root: str) -> None:
        self.root = root.rstrip("/")

    @property
    def base(self) -> str:
        """The directory every pinned tree on this host lives in."""
        return f"{self.root}/{SOURCES}"

    def path(self, key: str) -> str:
        """Where the tree `key` names is pinned, whether or not it has been materialised yet.

        Pure path arithmetic on purpose: a dispatch has to render the job script that runs from
        this directory before it opens the connection that creates it.
        """
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
            or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
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

        A key already pinned is verified without rebuilding its tree. Its declared results path,
        its needs and its wrapper are linked on every dispatch all the same, since those belong
        to the dispatch rather than to the tree and two batches off one commit routinely declare
        different ones.

        agent: the host's agent, which builds the tree under the host's pin lock.
        key: the tree's identity, the source's key.
        image: what the snapshot copies out of the mirror and what it links back.
        results: the dispatch's declared results path, linked back to the mirror so what the job
            writes there is what a later pull brings home; empty for a dispatch that declared
            none.
        prefix: the immutable environment this tree activates, named by content. Written once,
            when the tree is built, and never repointed: the environment belongs to the tree the
            way its code does, so a wave queued against it keeps it however often the workspace
            re-solves afterwards. Empty leaves the tree reaching the mirror's own environment,
            which is what a workspace with no addressed prefixes still does.
        environment: the environment `prefix` belongs to, which is the directory inside the
            generated tree the link is written into.
        script: staged generated wrapper to freeze and verify before submission; empty for a
            direct remote command whose file is not shipped.
        commit / digest: the dispatching tree's own provenance, written into the stamp beside
            the key so the tree on the host says which commit it is and what its content hashed
            to. A mirror carries no history, so this file is the only place on that machine
            where either can be read.
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
