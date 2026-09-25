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
# file when the universe root is digested.
BASELINES, SOURCES = "baselines", "*.py"

# Every digest's width in bytes: a fingerprint a person can compare by eye in a receipt column. It
# accelerates the question whether two trees are the same and never settles it.
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


def parsed(manifest: str) -> list[Row]:
    """The rows of a source listing, one `path<TAB>blob<TAB>status` line each."""
    return [
        Row(path=path, blob=blob, status=Status(status))
        for path, blob, status in (line.split("\t") for line in manifest.splitlines())
    ]


def _built(name: str) -> bool:
    """Whether one relative path is build output rather than source a digest should read."""
    return any(part.startswith(".") or part == "__pycache__" for part in name.split("/"))


def digest_of(directory: Path, pattern: str = "*") -> str:
    """One digest over every file under `directory` matching `pattern`, in relative-path order.

    The relative path is folded in beside the bytes, so a moved file changes the digest too.
    Caches and dot-directories are skipped as build output that would digest one tree two ways on
    two machines. Empty for a missing or empty directory, rather than the digest of nothing.
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
    """One registration row's digest, over canonical JSON so key order cannot move it.

    A lane's gate rides on the receipt as this: the exact row a verdict was scored against.
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
        rows = parsed(manifest)
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
    closure = tree.archive(manifest).with_suffix(".tsv")
    closure.write_text(manifest, encoding="utf-8")
    return Source(digest=captured.digest, closure=str(closure), root=tree.root)


def installed(name: str) -> str:
    """The installed version behind one logical package name, or `absent`.

    A platform may ship the import under another distribution name (`triton-windows` for
    `triton`), so the import-to-distribution index keeps the receipt schema platform-independent.
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
    name: the human name, a display column and never an identity.
    driver: the HOST DRIVER version, `610.57.04` shaped. It once held `cudaDriverGetVersion()`,
        the maximum CUDA a driver supports, so a generation of receipts stamped `13.3` and no
        driver at all, as three reviews found separately on 2026-08-29 (`fprev_recovery` 7c,
        `recovery_cost` 7d, `accuracy_selection` 6e).
    runtime: the compute runtime version beside it, `13.3` shaped, the CUDA one on an NVIDIA host.
    capability: the architecture key a kernel dispatches on.
    probed: `found`, `absent` on a host that carries no device, `failed` when the probe broke.
    detail: what the probe said when it broke, empty otherwise.
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

    A probe is a whole vendor stack whose failures are not ours to enumerate, so any failure is
    reported as `failed` rather than mistaken for an absent device or taking the session down.
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
            (self.source.root / row.path).resolve(): row.blob for row in parsed(manifest)
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
        admitted = expected is not None and blob_of(lane) == expected
        return Admissibility.ADMISSIBLE if admitted else Admissibility.UNRECORDED

    def baselines(self, node: str) -> str:
        """The digest of one claim's registered `baselines/` rows, empty where it registers none.

        A gate is pre-registered only if the rows it reads existed before the reading, so the
        digest rides on every receipt of the claim.
        """
        return digest_of(self.root / node / BASELINES)
