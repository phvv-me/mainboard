"""Source identity from the files that run, independent of version control.

What ran is kept as the listing (`closures`) and each listed file's bytes once per content
(`blobs`) in the workspace lake, so any dispatched tree can be rebuilt exactly, whichever of
the workspace's repositories a file belongs to and whether or not it was ever committed.
"""

import hashlib
import re
from datetime import UTC, datetime
from enum import StrEnum, auto
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

from patos import FrozenModel
from sqlalchemy import select

from ..core.errors import MissionError
from ..core.project import Project
from ..manifest.loading import load
from ..manifest.schema.workspace import DATA
from ..state import schema
from ..state.blobs import Blobs
from ..state.lake import Lake, insert
from .sync import GitignoreFilter

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    import duckdb

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]")


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


def parsed(manifest: str) -> list[Row]:
    """The rows of a listing `listing` wrote."""
    return [
        Row(path=path, blob=blob, status=Status(status))
        for path, blob, status in (line.split("\t") for line in manifest.splitlines() if line)
    ]


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
        manifest = Project().manifest(self.root)
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

    def archive(self, manifest: str) -> str:
        """Keep retrievable source bytes before dispatch, not merely their hashes; return the
        listing's digest.

        Each listed file's bytes land in the lake's `blobs` once per content, a file unchanged
        since any earlier dispatch costing nothing, and the listing in `closures`, all in one
        commit. A file that changed since it was listed is refused rather than kept under the
        wrong digest.
        """
        rows = parsed(manifest)
        digest = hashlib.sha256(manifest.encode()).hexdigest()
        lake = Lake.at(self.root).ready()
        blobs = Blobs(lake)
        closure = digest[:12]

        def keep(connection: duckdb.DuckDBPyConnection) -> None:
            held = blobs.held(connection, {row.blob for row in rows})
            blobs.stage(
                connection,
                {row.blob: self.root / row.path for row in rows if row.blob not in held},
            )
            kept = schema.closures
            listed = lake.execute(
                connection, select(kept.c.closure).where(kept.c.closure == closure).limit(1)
            ).fetchone()
            if listed is None:
                stamp = datetime.now(UTC)
                insert(
                    connection,
                    schema.closures,
                    [
                        {
                            "ts": stamp,
                            "closure": closure,
                            "path": row.path,
                            "blob": row.blob,
                            "status": str(row.status),
                        }
                        for row in rows
                    ],
                )

        lake.transact(keep)
        return digest

    def restore(self, digest: str, into: Path) -> list[Path]:
        """Write the tree whose listing has `digest` under `into`, byte for byte, from the lake.

        Raises MissionError when the lake holds no such listing or misses one of its files.
        """
        lake = Lake.at(self.root).current()
        with lake.open() as connection:
            kept = schema.closures
            listing = (
                select(kept.c.path, kept.c.blob)
                .where(kept.c.closure == digest[:12])
                .order_by(kept.c.path)
            )
            rows = lake.execute(connection, listing).fetchall()
            payloads = Blobs(lake).read(connection, {blob for _, blob in rows})
        if not rows:
            raise MissionError(f"the lake keeps no source listing {digest[:12]}")
        written: list[Path] = []
        for path, blob in rows:
            if blob not in payloads:
                raise MissionError(f"the lake keeps no bytes for {path} ({blob[:12]})")
            target = into / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payloads[blob])
            written.append(target)
        return written
