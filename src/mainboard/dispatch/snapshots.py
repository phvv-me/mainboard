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

import re
import shlex
import shutil
import subprocess  # ruff:ignore[suspicious-subprocess-import]  reason=asks the host's own pueue what is queued, fixed argv since=2026-09-29
import time
from abc import ABC, abstractmethod
from contextlib import suppress
from functools import reduce
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

from patos import FrozenModel, FrozenOpenModel

from ..core.project import Project
from ..core.retention import Retention, RetentionPins
from ..jobs.pins import staging
from .agent import AgentRefused
from .agent.program import WRAPPERS, ScopeSpec, reserved
from .spool import Spool, Submission, spool_path

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

    from .agent import Agent
    from .agent.program import ImageSpec
    from .state.cache import RunRecord

# A snapshot unused this long, not among the newest `_KEPT_SNAPSHOTS` and named by nothing the
# inventory knows goes: a week of edit-dispatch cycles was 15 GB.
_SNAPSHOT_DAYS = 7
_KEPT_SNAPSHOTS = 3
# A snapshot key as it appears in a job script or a task's command.
_KEY = re.compile(rf"/{RetentionPins.SOURCES}/([^/\s'\"]+)")


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
    """Whether `path` would replace one of a snapshot's own control files, under any name."""
    return path == "." or reserved(path)


def _cwd(submission: Path) -> list[str]:
    """The tree the held submission at `submission` runs from."""
    return [Submission.model_validate_json(submission.read_text(encoding="utf-8")).cwd]


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
        file. Frozen as `CLOSURE`, which the runner reads through `MB_CLOSURE` rather than
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
            "staging": staging(),
            "live": live,
        }


class Snapshots:
    """The pinned source trees on one host, under `{root}/{state}/sources/`.

    Source files share inodes with the mirror; directories and frozen metadata add storage.
    Hardlinking also makes the pin correct: the mirror replaces a changed file by renaming a new
    one over it, so only the mirror's entry moves to the new inode. Every pin also publishes the
    tree as a root to keep (`Retention`), so pruning removes only what an inventory has fully
    accounted for. Watch inode quotas.

    root: the host's workspace root, the mirror every snapshot is taken from and links back to.
    out: the generated-state directory name under `root`; the dispatching workspace's own when
        empty, since a remote root is arithmetic and no local directory says what it is called.
    """

    def __init__(self, root: str, *, out: str = "") -> None:
        self.root = root.rstrip("/")
        self.retention = RetentionPins(Path(self.root), out or Project().out_dir())

    @property
    def base(self) -> str:
        return str(self.retention.base)

    def inventory(self, *, busy: str | None, registry: Sequence[RunRecord] = ()) -> Retention:
        """Collect every root the schedulers, held spool, staged scripts and registry name.

        What cannot be read is an `unknown`, which keeps every root. The host's own pins join at
        pruning, under the lock that orders them against the agent.

        busy: text naming the tasks pueue lists (`queued_here`), None when it could not be asked.
        registry: every run the caller's registry holds, whatever its verdict. A settled run's
            source stays the proof of its evidence, and a run whose handle was lost may be live.
        """
        spool = Spool(Path(spool_path(self.root)))
        observed = (
            self._queued(busy),
            Retention(proofs=frozenset(self.path(run.source) for run in registry if run.source)),
            self._named(spool.inbox, ".json", _cwd),
            # A claim names the tree of its submission; one whose submission is gone is unknown.
            self._named(spool.claims, ".json", lambda claim: _cwd(spool.inbox / claim.name)),
            self._named(Path(self.base).parent / "jobs", ".sh", self._scripted),
        )
        return reduce(Retention.including, observed, Retention())

    def _queued(self, busy: str | None) -> Retention:
        if busy is None:
            return Retention(unknown=("scheduler inventory unavailable",))
        return Retention(sources=frozenset(self.path(key) for key in _KEY.findall(busy)))

    def _scripted(self, script: Path) -> list[str]:
        """The trees a staged job script runs from."""
        return [self.path(key) for key in _KEY.findall(script.read_text(encoding="utf-8"))]

    @staticmethod
    def _named(directory: Path, suffix: str, roots: Callable[[Path], Iterable[str]]) -> Retention:
        """Collect the trees `roots` finds in each `suffix` file of `directory`.

        A directory that is not there names nothing; one holding a file that cannot be read or
        understood leaves ownership unknown, never empty.
        """
        if not directory.is_dir():
            return Retention()
        try:
            named = [
                root
                for file in directory.iterdir()
                if file.suffix == suffix and file.name[0] != "."
                for root in roots(file)
            ]
        except OSError, ValueError:
            return Retention(unknown=(f"unreadable references in {directory}",))
        return Retention(sources=frozenset(named))

    def path(self, key: str) -> str:
        """Where `key` is pinned, materialised or not: pure arithmetic, because a dispatch renders
        the job script that runs from it before opening the connection that creates it."""
        return f"{self.base}/{key}"

    def prune(self, inventory: Retention) -> list[str]:
        """Remove, on this host, the snapshots nothing needs any more, and name them.

        Kept: the newest few, any used (pinned or re-pinned) within `_SNAPSHOT_DAYS`, and any
        root `inventory` or this host's pins hold. Only the tree goes: its files are hardlinks
        and its links point into the mirror, which stays whole.

        inventory: what `inventory` found; an unknown in it, or in the pins, removes nothing.
        """
        return self.retention.swept(inventory, self._sweep)

    def _sweep(self, inventory: Retention) -> list[str]:
        try:
            trees = sorted(
                (
                    entry
                    for entry in Path(self.base).iterdir()
                    if not entry.is_symlink() and entry.is_dir() and entry.name[0] != "."
                ),
                key=lambda entry: entry.stat().st_mtime,
                reverse=True,
            )
        except OSError:
            return []
        horizon = time.time() - _SNAPSHOT_DAYS * 86400
        doomed = [
            tree
            for tree in trees[_KEPT_SNAPSHOTS:]
            if tree.stat().st_mtime < horizon and not inventory.protects(tree)
        ]
        for tree in doomed:
            shutil.rmtree(tree, ignore_errors=True)
        return [tree.name for tree in doomed]

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
                        "out": Project().out_dir(),
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


class _Task(FrozenOpenModel):
    """The two fields of a `pueue status --json` task that say what runs and where."""

    command: str
    path: str

    def text(self) -> str:
        """The command, its directory and each wrapper script it names that still exists."""
        texts = [self.command, self.path]
        for script in (word for word in shlex.split(self.command) if word.endswith(".sh")):
            with suppress(FileNotFoundError):
                texts.append(Path(self.path, script).read_text(encoding="utf-8"))
        return "\n".join(texts)


class _Queue(FrozenOpenModel):
    tasks: dict[str, _Task]


def queued_here() -> str | None:
    """Text naming every task this host's pueue lists, finished ones too, and their scripts.

    A finished task stays listed until the center cleans it, so completion alone retires nothing.
    None when pueue, or a script it names, cannot be read: nothing is then known to be unused.
    """
    pueue = shutil.which("pueue")
    if pueue is None:
        return None
    try:
        said = subprocess.run(
            [pueue, "status", "--json"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=20,
            check=True,
        ).stdout
        return "\n".join(task.text() for task in _Queue.model_validate_json(said).tasks.values())
    except OSError, subprocess.SubprocessError, ValueError:
        return None
