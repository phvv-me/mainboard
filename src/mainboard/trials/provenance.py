# WHERE A READING WAS TAKEN, PROBED ONCE AND STAMPED ON EVERY RECEIPT OF A RUN.
#
# Everything here is DERIVED. A fact a test has to retype is a fact a test will eventually retype
# wrong, and two harnesses that each spell their own provenance produce two receipts that cannot
# be compared on the axis that matters most, so this is the one probe and the one shape.
#
# A COMMIT NAMES A TREE AND A DIRTY COMMIT NAMES NOTHING. The generation of receipts before this
# one carried a SHORT hash and a dirty boolean, and a review on 2026-08-29 read a whole program's
# evidence off them: every claimed row said `d38692f` and `worktree_dirty: true`, so two different
# trees, hours and edits apart, wrote receipts that are the same string. The claim behind them
# could not be reconstructed from anything the receipt held. What closes that is not a longer hash
# but three facts instead of one: the FULL commit, the COMMITTED TREE it resolves to, and a digest
# of the source files ACTUALLY ON DISK, untracked ones included, since untracked files are exactly
# where two dirty trees at one commit differ.
#
# AND A TREE NOBODY CAN IDENTIFY IS INADMISSIBLE RATHER THAN FORBIDDEN. Refusing to collect on a
# dirty tree would make the tool useless for the work it is most used for, which is trying
# something and looking at the number. So a run on a moving tree still runs, still writes, and
# still prints, and every row it writes SAYS SO in a typed field a coverage or verdict query
# filters on. Scratch work stays possible and stops counting as evidence, which is the whole of
# the rule: a claim needs a clean tree, an experiment does not.
#
# THE CARD IS IDENTIFIED BY ITS UUID AND NOT BY ITS NAME. A name is a model number: two identical
# 4090s in one box answer `NVIDIA GeForce RTX 4090` to the same question, so a lane satisfied on
# the first would read complete on the second and publish one card's rows twice. The UUID is the
# device's identity and is what `card` carries, which makes it the coverage axis; the name rides
# beside it in `card_name` because a table has to print something a reader recognises. A provider
# that exposes no UUID falls back to the name, which is the old behaviour and still better than an
# empty coordinate every card would share.
#
# A HOST WITH NO DEVICE IS A HOST, NOT AN ERROR. The reference this was lifted from refuses to
# stamp a receipt where `nvidia-smi` is absent, which is right for a lab whose every claim reads a
# GPU and wrong for a subsystem that also serves a pure-theory run and this tool's own CI. An
# absent card is recorded as the empty card and the refusal is left to whoever actually needs one.
#
# BUT AN ABSENT CARD AND A BROKEN PROBE ARE NOT THE SAME EMPTY STRING. One says this run measured
# no device and the other says nobody knows what it measured, and a coverage question that cannot
# tell them apart will hand a theory host's cell to a machine whose probe simply fell over. So the
# outcome is recorded beside the value, and a probe that raises is caught HERE and reported rather
# than taking a whole session down, the same translation `mainboard.lab.gates.Gate.evaluate`
# already makes for a precondition check that breaks.
#
# A MIRROR HAS THE SOURCE AND NOT THE HISTORY. A dispatch rsyncs the working tree onto a host and
# the repository stays behind, so `git rev-parse` on a remote card answers nothing and every
# receipt a dispatched job writes could name no commit. The dispatcher declares its own HEAD in
# `MAINBOARD_SOURCE` instead, spelled as `git describe --always --dirty` spells it, and a row
# written under it says `mirrored` so a declared commit is never read as a probed one. Refusing
# outright would say no remote host may ever take a reading, which is the wrong answer to a
# cross-architecture question. A mirror still digests the source it was handed, which is the one
# identity that survives the trip, and a dispatcher declaring `-dirty` lands inadmissible for the
# same reason a local dirty tree does.

import json
import os
import platform
from enum import StrEnum, auto
from hashlib import blake2b
from importlib.metadata import PackageNotFoundError, packages_distributions, version
from typing import TYPE_CHECKING

from patos import FrozenModel

from ..dispatch.shared import git
from ..probe.machine import Machine
from .coverage import Probed

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

    from pydantic import JsonValue

# The variable a mirrored run reads to learn which source it is, set by whatever dispatched it.
SOURCE_VAR = "MAINBOARD_SOURCE"

# What a claim's registered rows live under, one directory per node, and what counts as a source
# file when the universe root is digested. Both are named here because the digest is taken here.
BASELINES, SOURCES = "baselines", "*.py"

# How many bytes every digest here is. Sixteen is a fingerprint a person can read in a receipt
# column and compare by eye, and it is a DIGEST rather than an identity: it accelerates the
# question of whether two trees are the same and never settles it.
WIDTH = 16


class Admissibility(StrEnum):
    """Whether one row's producing tree can be identified, which is what makes it evidence.

    ADMISSIBLE: the tree was clean and the lane's own file was committed in it.
    DIRTY: the tree carried uncommitted changes, so two runs at this commit are two trees.
    UNTRACKED: the lane's file is in no commit, so the named commit does not contain the test.
    UNRECORDED: the row was written before this field existed and can prove none of the above.
    """

    ADMISSIBLE = auto()
    DIRTY = auto()
    UNTRACKED = auto()
    UNRECORDED = auto()


def _built(name: str) -> bool:
    """Whether one relative path is build output rather than source a digest should read."""
    return any(part.startswith(".") or part == "__pycache__" for part in name.split("/"))


def digest_of(directory: Path, pattern: str = "*") -> str:
    """One digest over every file under `directory` matching `pattern`, in relative-path order.

    The relative path is folded in beside the bytes, so a file that MOVED changes the digest as
    surely as a file that changed. Caches and dot-directories are skipped because they are build
    output rather than source and would digest the same tree differently on two machines. Empty
    for a directory that does not exist or holds nothing, which says `there is no such thing`
    rather than handing back the digest of nothing at all.
    """
    if not directory.is_dir():
        return ""
    found = (path.relative_to(directory).as_posix() for path in directory.rglob(pattern))
    names = sorted(name for name in found if not _built(name) and (directory / name).is_file())
    if not names:
        return ""
    running = blake2b(digest_size=WIDTH)
    for name in names:
        running.update(name.encode())
        running.update(b"\0")
        running.update(blake2b((directory / name).read_bytes(), digest_size=WIDTH).digest())
    return running.hexdigest()


def digested(payload: JsonValue) -> str:
    """One registration row's digest, over its canonical JSON so key order cannot move it.

    This is what a lane's GATE rides on the receipt as: the exact committed row a verdict was
    scored against, fingerprinted where it is read rather than described in prose afterwards.
    """
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return blake2b(canonical.encode(), digest_size=WIDTH).hexdigest()


class Source(FrozenModel):
    """Which tree took a reading, named by its history and pinned by its bytes.

    commit: the FULL commit id, never the short one. A short hash is a display convenience that
        stopped being one the day a program's whole evidence base named the same seven characters.
    tree: the committed tree object the commit resolves to, empty on a mirror with no repository.
    digest: the canonical digest of the source files actually on disk, untracked ones included,
        which is the only identity a dirty tree or a mirror has.
    dirty: whether the working tree carried uncommitted changes.
    mirrored: whether the commit was declared by a dispatcher rather than probed from a repo.
    """

    commit: str
    tree: str = ""
    digest: str = ""
    dirty: bool = False
    mirrored: bool = False

    @property
    def admissibility(self) -> Admissibility:
        """Whether readings taken from this tree are evidence or scratch work."""
        return Admissibility.DIRTY if self.dirty else Admissibility.ADMISSIBLE


def source(repo: Path, *, read: Callable[..., str] = git) -> Source:
    """The commit, the committed tree, the cleanliness and the mirror flag of one working tree.

    repo: the working tree the reading was taken from.
    read: the git reader, the local `git` by default and a stand-in under test.
    """
    head = read("-C", str(repo), "rev-parse", "HEAD")
    if head:
        return Source(
            commit=head,
            tree=read("-C", str(repo), "rev-parse", "HEAD^{tree}"),
            dirty=bool(read("-C", str(repo), "status", "--porcelain")),
        )
    declared = os.environ.get(SOURCE_VAR, "")
    if not declared:
        raise RuntimeError(
            f"{repo} is not a git working tree and {SOURCE_VAR} is unset, so this reading could "
            f"name no source; a dispatched job sets {SOURCE_VAR} to the dispatcher's own "
            "`git describe --always --dirty`"
        )
    return Source(
        commit=declared.removesuffix("-dirty"), dirty=declared.endswith("-dirty"), mirrored=True
    )


def installed(name: str) -> str:
    """The installed version behind one logical package name, or `absent`.

    A platform may publish the same import from a differently named distribution, as
    `triton-windows` does for the `triton` package. The import-to-distribution index keeps the
    receipt schema platform-independent without hard-coding either platform's spelling.
    """
    try:
        return version(name)
    except PackageNotFoundError:
        found: set[str] = set()
        for distribution in packages_distributions().get(name, ()):
            try:
                found.add(version(distribution))
            except PackageNotFoundError:
                continue
        return found.pop() if len(found) == 1 else "absent"


class Card(FrozenModel):
    """The device a reading was taken on, and whether the probe actually found one.

    id: the device UUID, the coverage identity, falling back to the name where none is exposed.
    name: the human name, which is a display column and never an identity.
    driver: the HOST DRIVER version the reading ran under, `610.57.04` shaped.
    runtime: the compute runtime version beside it, `13.3` shaped, the CUDA one on an NVIDIA host.
    capability: the architecture key a kernel dispatches on.
    probed: `found`, `absent` on a host that carries no device, `failed` when the probe broke.
    detail: what the probe said when it broke, empty otherwise.

    THE TWO VERSION FIELDS ARE TWO FACTS AND THE RECEIPT USED TO CARRY ONE OF THEM TWICE. `driver`
    held `cudaDriverGetVersion()`, the maximum CUDA a driver supports, under a name that promised
    the driver, so a generation of receipts stamped `13.3` on a host whose driver is `610.57.04`
    and carried no driver version at all. Three independent reviews on 2026-08-29
    (`fprev_recovery` 7c, `recovery_cost` 7d, `accuracy_selection` 6e) found it separately, which
    is what a field whose name and content disagree costs.
    """

    id: str = ""
    name: str = ""
    driver: str = ""
    runtime: str = ""
    capability: str = ""
    probed: Probed = Probed.ABSENT
    detail: str = ""


def card_of(machine: Machine) -> Card:
    """The first visible device, an empty card where there is none, and why either way.

    A probe is a whole vendor stack behind one attribute, so the set of ways it can fail is not
    ours to enumerate. What matters is that a broken one is REPORTED rather than mistaken for an
    absent device or allowed to take the session down, so the receipt says the machine is unknown
    and a reader can act on that.
    """
    try:
        cards = machine.gpus
    except Exception as error:
        return Card(probed=Probed.FAILED, detail=str(error))
    if not cards:
        return Card(probed=Probed.ABSENT)
    found = cards[0]
    runtime = found.runtime_version
    return Card(
        id=found.uuid or found.label,
        name=found.label,
        driver=found.driver,
        runtime=".".join(str(part) for part in runtime) if runtime else "",
        capability=found.arch_key,
        probed=Probed.FOUND,
    )


class Preflight:
    """Everything known about the producing tree and machine BEFORE a single trial is collected.

    Taken once, at the top of a run, because that is the only moment at which the answer is about
    the state the run is ABOUT TO measure from rather than the state a lane has already moved. It
    is also what makes admissibility cheap: the tracked set is read once and every lane after that
    is a membership test rather than another `git` process.

    root: the universe root whose source files are digested and whose lanes are checked.
    repo: the working tree the commit, the tree and the tracked set are read from.
    probed: the distributions whose version can move a reading here.
    machine: the probed machine, this host's when omitted.
    read: the git reader, the local `git` by default and a stand-in under test.
    """

    def __init__(
        self,
        root: Path,
        repo: Path,
        *,
        probed: Sequence[str] = (),
        machine: Machine | None = None,
        read: Callable[..., str] = git,
    ) -> None:
        self.root = root
        self.source = source(repo, read=read)
        self.digest = digest_of(root, SOURCES)
        listed = (
            ""
            if self.source.mirrored
            else read("-C", str(repo), "ls-files", "-z", "--", str(root))
        )
        # None rather than an empty set, because a mirror carries no history to ask and an
        # unanswerable question must not read as the answer `nothing here is tracked`.
        self.tracked = (
            None
            if self.source.mirrored
            else frozenset((repo / name).resolve() for name in listed.split("\0") if name)
        )
        self.card = card_of(machine or Machine())
        self.versions: dict[str, JsonValue] = {name: installed(name) for name in probed}

    @property
    def admissibility(self) -> Admissibility:
        """Whether this run's readings can be evidence at all, before any lane is looked at."""
        return self.source.admissibility

    @property
    def stamp(self) -> dict[str, JsonValue]:
        """The host, the card and the tree, so no reading can be read on the wrong machine."""
        return {
            "host": platform.node(),
            "card": self.card.id,
            "card_probed": str(self.card.probed),
            "card_name": self.card.name,
            "card_detail": self.card.detail,
            "driver": self.card.driver,
            "runtime": self.card.runtime,
            "capability": self.card.capability,
            "commit": self.source.commit,
            "tree": self.source.tree,
            "source_digest": self.digest,
            "worktree_dirty": self.source.dirty,
            "mirrored": self.source.mirrored,
            "versions": self.versions,
        }

    def admits(self, lane: Path) -> Admissibility:
        """Whether one lane's own file can be identified in the tree this run was taken from.

        The run-wide answer comes first, because a dirty tree makes every lane in it unidentifiable
        whatever git thinks of any one file. A mirror is not asked, since the repository it was
        copied from is not here and an unanswerable question is not a failed one.

        lane: the file the trial was collected from.
        """
        if self.source.dirty:
            return Admissibility.DIRTY
        if self.tracked is None or lane.resolve() in self.tracked:
            return Admissibility.ADMISSIBLE
        return Admissibility.UNTRACKED

    def baselines(self, node: str) -> str:
        """The digest of one claim's registered rows, empty where the claim registers none.

        A gate is only pre-registered if the rows it reads existed before the reading did, so the
        whole `baselines/` directory rides on every receipt of the claim that owns it.

        node: which claim to digest, the universe root itself for a flat universe.
        """
        return digest_of(self.root / node / BASELINES)
