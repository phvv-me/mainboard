"""Capture the exact source and machine before acquisition, without version control."""

import json
import os
import platform
from enum import StrEnum, auto
from hashlib import blake2b, sha256
from importlib.metadata import PackageNotFoundError, packages_distributions, version
from pathlib import Path
from typing import TYPE_CHECKING

from patos import FrozenModel

from ..dispatch.provenance import Row, SourceTree, Status, blob_of, listing
from ..dispatch.shared import CLOSURE_VAR, DIGEST_VAR
from ..probe.machine import Machine
from .coverage import Probed

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pydantic import JsonValue

# What a claim's registered rows live under, one directory per node, and what counts as a source
# file when the universe root is digested. Both are named here because the digest is taken here.
BASELINES, SOURCES = "baselines", "*.py"

# How many bytes every digest here is. Sixteen is a fingerprint a person can read in a receipt
# column and compare by eye, and it is a DIGEST rather than an identity: it accelerates the
# question of whether two trees are the same and never settles it.
WIDTH = 16


class Admissibility(StrEnum):
    """Whether one row's producing tree can be identified, which is what makes it evidence.

    ADMISSIBLE: the lane belongs to the verified content snapshot.
    DIRTY: historical rejection label, preserved when reading old receipts.
    UNTRACKED: historical rejection label, preserved when reading old receipts.
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
    """The verified content snapshot used for a reading.

    closure: path to the authenticated listing, relative to the workspace when local.
    mirrored: whether a dispatcher supplied the snapshot.
    root: the workspace against which listing paths resolve, not a nested trial project.
    """

    digest: str
    closure: str
    root: Path
    mirrored: bool = False

    @property
    def admissibility(self) -> Admissibility:
        """Only captured and verified bytes qualify as identified source."""
        return (
            Admissibility.ADMISSIBLE if self.digest and self.closure else Admissibility.UNRECORDED
        )


def source(repo: Path) -> Source:
    """Verify a dispatched snapshot or preserve a local source bundle before acquisition."""
    declared = os.environ.get(CLOSURE_VAR, "")
    expected = os.environ.get(DIGEST_VAR, "")
    if declared:
        closure = Path(declared).resolve()
        tree = SourceTree(closure.parent)
        if not repo.resolve().is_relative_to(tree.root):
            raise RuntimeError("trial project is outside the captured source workspace")
        manifest = closure.read_text(encoding="utf-8")
        if not expected or sha256(manifest.encode()).hexdigest() != expected:
            raise RuntimeError("source listing does not match the declared content digest")
        rows = [
            Row(path=p, blob=b, status=Status(s))
            for p, b, s in (line.split("\t") for line in manifest.splitlines())
        ]
        verified, _ = tree.seal(
            [row.path for row in rows],
            built=[row.path for row in rows if row.status is Status.BUILT],
        )
        if verified.digest != expected:
            raise RuntimeError("source bytes differ from the captured bundle")
        return Source(digest=expected, closure=str(closure), root=tree.root, mirrored=True)
    tree = SourceTree(repo)
    captured, rows = tree.seal(tree.kept("."))
    manifest = listing(rows)
    archive = tree.archive(manifest)
    closure = archive.with_suffix(".tsv")
    closure.write_text(manifest, encoding="utf-8")
    return Source(digest=captured.digest, closure=str(closure), root=tree.root)


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
    """Capture the source once, then check lane membership and bytes before each trial."""

    def __init__(
        self,
        root: Path,
        repo: Path,
        *,
        probed: Sequence[str] = (),
        machine: Machine | None = None,
    ) -> None:
        self.root = root
        self.source = source(repo)
        self.digest = self.source.digest
        manifest = Path(self.source.closure).read_text(encoding="utf-8")
        self.captured = {
            (self.source.root / p).resolve(): b
            for p, b, _ in (line.split("\t") for line in manifest.splitlines())
        }
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
            "source_digest": self.digest,
            "closure": self.source.closure,
            "mirrored": self.source.mirrored,
            "versions": self.versions,
        }

    def admits(self, lane: Path) -> Admissibility:
        """Whether the lane's current bytes belong to the captured source."""
        expected = self.captured.get(lane.resolve())
        if expected is None:
            return Admissibility.UNRECORDED
        return Admissibility.ADMISSIBLE if blob_of(lane) == expected else Admissibility.UNRECORDED

    def baselines(self, node: str) -> str:
        """The digest of one claim's registered rows, empty where the claim registers none.

        A gate is only pre-registered if the rows it reads existed before the reading did, so the
        whole `baselines/` directory rides on every receipt of the claim that owns it.

        node: which claim to digest, the universe root itself for a flat universe.
        """
        return digest_of(self.root / node / BASELINES)
