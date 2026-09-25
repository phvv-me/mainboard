"""Source identity from the files that run, independent of version control."""

import hashlib
import re
import shutil
import time
from enum import StrEnum, auto
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING
from zipfile import ZIP_DEFLATED, ZipFile

from patos import FrozenModel

from ..core.errors import MissionError
from ..core.project import Project
from ..manifest.loading import load
from ..manifest.schema.workspace import DATA
from .sync import GitignoreFilter

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]")
_ARCHIVE_LISTING = ".mainboard-source-listing.tsv"
# How old a partial archive, or a temporary folder an older release archived in, must be before
# it counts as left by a killed process rather than one still writing.
_STALE_SECONDS = 86_400


class Status(StrEnum):
    """File kind; historical version-control labels remain readable, never gate execution."""

    SOURCE = auto()
    BUILT = auto()
    CLEAN = auto()
    MODIFIED = auto()
    UNTRACKED = auto()
    IGNORED = auto()
    UNVERSIONED = auto()


class Source(FrozenModel):
    """The content identity and snapshot key of one source bundle.

    commit: historical metadata only; new bundles leave it empty.
    """

    identity: str
    key: str
    digest: str = ""
    commit: str = ""


class Row(FrozenModel):
    """A relative source path, its raw SHA-256 digest, and its file kind."""

    path: str
    blob: str
    status: Status = Status.SOURCE


def blob_of(path: Path) -> str:
    """Hash the actual file bytes without a repository or external executable."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def listing(rows: Iterable[Row]) -> str:
    """The portable source manifest, one path, digest and kind per line."""
    return "".join(f"{row.path}\t{row.blob}\t{row.status}\n" for row in rows)


def registered(node: Path, rows: Sequence[Row], *, root: Path) -> None:
    """Require the registration bytes captured before acquisition, not a commit."""
    relative = node.relative_to(root).as_posix()
    row = next((item for item in rows if item.path == relative), None)
    if row is None:
        raise MissionError(f"{relative} must be captured before this job is acquired")
    if blob_of(node) != row.blob:
        raise MissionError(f"{relative} changed after Mainboard prepared the job")


def named(identity: str) -> str:
    """A safe snapshot-directory component."""
    return _UNSAFE.sub("-", identity)[:96].lstrip(".") or "source"


class SourceTree:
    """Discover, fingerprint and preserve source without consulting version control."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.filter = GitignoreFilter(root)

    def kept(self, directory: str) -> list[str]:
        """Files under an explicit source root, honoring optional ignore files and secrets."""
        return self.filter.files([directory])

    def sources(self) -> list[str]:
        """The workspace's source: every file a host could be sent, less its `[workspace] data`.

        A dataset a trial reads is pinned through `needs` or `resources`, never archived as
        source; a host mirror still ships whatever its own sync include names.
        """
        manifest = self.root / Project().manifest
        data = load(manifest).workspace.data if manifest.is_file() else DATA
        return self.filter.files(
            sorted(entry.name for entry in self.root.iterdir()), excluded=data
        )

    def seal(self, files: Sequence[str], *, built: Sequence[str] = ()) -> tuple[Source, list[Row]]:
        """Identify exactly these files as they stand on disk, including newly created files."""
        self.filter.validate_sources(files)
        marked = frozenset(built)
        rows = []
        for path in sorted(set(files)):
            relative = PurePosixPath(path)
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or any(c in path for c in "\t\r\n")
            ):
                raise MissionError(f"invalid source path: {path!r}")
            if any(part in {".git", ".env"} for part in relative.parts):
                raise MissionError(f"private control file cannot be source: {path}")
            if not (self.root / path).resolve().is_relative_to(self.root):
                raise MissionError(f"source escapes workspace: {path}")
            rows.append(
                Row(
                    path=path,
                    blob=blob_of(self.root / path),
                    status=Status.BUILT if path in marked else Status.SOURCE,
                )
            )
        digest = hashlib.sha256(listing(rows).encode()).hexdigest()
        return Source(identity=f"sha256:{digest}", key=f"sha256-{digest}", digest=digest), rows

    def archive(self, manifest: str) -> Path:
        """Keep retrievable source bytes locally before dispatch, not merely their hashes.

        The zip is written as `<digest>.zip.partial` and renamed whole, so a killed archival
        leaves one partial its retry truncates, and archiving sweeps what is a day stale.
        """
        digest = hashlib.sha256(manifest.encode()).hexdigest()
        target = self.root / Project().out_dir / "source-archives" / f"{digest}.zip"
        rows = [
            Row(path=p, blob=b, status=Status(s))
            for p, b, s in (line.split("\t") for line in manifest.splitlines())
        ]
        target.parent.mkdir(parents=True, exist_ok=True)
        _swept(target.parent)
        if target.exists():
            with ZipFile(target) as archive:
                if archive.read(_ARCHIVE_LISTING).decode() != manifest or any(
                    hashlib.sha256(archive.read(row.path)).hexdigest() != row.blob for row in rows
                ):
                    raise MissionError(f"source archive verification failed: {target}")
            return target
        pending = target.with_name(f"{target.name}.partial")
        with ZipFile(pending, "w", compression=ZIP_DEFLATED) as archive:
            archive.writestr(_ARCHIVE_LISTING, manifest)
            for row in rows:
                payload = (self.root / row.path).read_bytes()
                if hashlib.sha256(payload).hexdigest() != row.blob:
                    raise MissionError(f"{row.path} changed before source archival")
                archive.writestr(row.path, payload)
        pending.replace(target)
        return target


def _swept(folder: Path) -> None:
    """Remove the partial archives and temporary folders killed archivals left a day ago."""
    stale = time.time() - _STALE_SECONDS
    for leftover in [*folder.glob("*.partial"), *folder.glob("source-*")]:
        if leftover.stat().st_mtime < stale:
            shutil.rmtree(leftover) if leftover.is_dir() else leftover.unlink()
