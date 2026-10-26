"""Keeping evidence: files read once and indexed at their paths, their bytes kept as objects."""

import hashlib
import mimetypes
import os
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from functools import cached_property, partial
from pathlib import Path, PurePosixPath

import duckdb
from patos import FrozenModel
from sqlalchemy import Select, func, or_, select

from ...core.errors import MissionError
from .. import schema
from ..constants import ALIAS, READERS, WINDOW_BYTES
from ..lake import Lake, Session, insert
from ..storage import blobs
from ..storage.blobs import Audit, Blobs
from ..storage.owned import OwnedFile, tree_of
from .models import Finalized, Identity, Kept, Located, located

# Where the read-through cache lives under the state directory.
_CACHE = "evidence"


_HELD_BYTES = 32 << 20


# The media types an evidence file's suffix does not name through `mimetypes`, and the bytes a
# Parquet file begins with, since a content-addressed object has no suffix at all.
_TYPES = {
    ".parquet": "application/vnd.apache.parquet",
    ".ndjson": "application/x-ndjson",
    ".jsonl": "application/jsonl",
    ".zst": "application/zstd",
}


_PARQUET = "application/vnd.apache.parquet"


_MAGIC = b"PAR1"


class _Entry(FrozenModel):
    """One evidence file on its way into the lake, read once: its bytes ride along unless it is
    too large to hold, in which case staging streams it from disk."""

    source: Path
    path: str
    sha256: str
    size: int
    media_type: str
    payload: bytes | None = None
    new: bool = False


def posix(relative: Path) -> str:
    """A relative path as the index spells it, the workspace root itself as empty."""
    spelled = relative.as_posix()
    return "" if spelled == "." else spelled


def within(root: Path, *, path: Path) -> str:
    """`path` as the index spells it under the workspace `root`.

    `path` is absolute, or relative to the current directory.

    Raises MissionError for a path outside the workspace.
    """
    absolute = Path(os.path.abspath(path))
    try:
        return posix(absolute.relative_to(root))
    except ValueError:
        raise MissionError(f"{absolute} lies outside the workspace at {root}") from None


def identity_of(metadata: os.stat_result) -> Identity:
    return metadata.st_dev, metadata.st_ino, metadata.st_size, metadata.st_mtime_ns


def listing(prefix: str = "") -> Select:
    """Select every indexed file at or under the workspace-relative `prefix`, all when empty."""
    kept = schema.evidence
    query = select(kept.c.path, kept.c.sha256, kept.c.size).order_by(kept.c.path)
    under = or_(kept.c.path == prefix, func.starts_with(kept.c.path, f"{prefix}/"))
    return query.where(under) if prefix else query


def windows[T](items: Sequence[T], weight: Callable[[T], int], limit: int) -> Iterator[list[T]]:
    """`items` in order, in windows weighing at most `limit` together.

    One item alone may be heavier, so whatever a window holds at once stays bounded.
    """
    window: list[T] = []
    size = 0
    for item in items:
        heft = weight(item)
        if window and size + heft > limit:
            yield window
            window, size = [], 0
        window.append(item)
        size += heft
    if window:
        yield window


def _media_type(path: Path, head: bytes) -> str:
    """What `path` holds, by its suffix, else by Parquet's leading magic in `head`.

    Anything else is raw bytes.
    """
    named = _TYPES.get(path.suffix) or mimetypes.guess_type(path.name)[0]
    return named or (_PARQUET if head == _MAGIC else "application/octet-stream")


def _labels(path: str) -> dict[str, str]:
    """The project, node and run a workspace-relative evidence path names.

    The project is the directory holding the first `datasets` or `experiments` directory (empty
    at the workspace root), the node the directory under `experiments`, and the run a `run=<id>`
    directory's id or the directory under `artifacts`; each empty where the path names none.
    """
    parts = PurePosixPath(path).parts[:-1]
    project = node = run = ""
    for index, part in enumerate(parts):
        following = parts[index + 1] if index + 1 < len(parts) else ""
        if part in {"datasets", "experiments"} and not (project or node):
            project = parts[index - 1] if index else ""
        if part == "experiments" and not node:
            node = following
        if part.startswith("run=") and not run:
            run = part.removeprefix("run=")
        if part == "artifacts" and not run:
            run = following
    return {"project": project, "node": node, "run": run}


class Keeping:
    """One workspace's evidence in its lake: the files kept and their paths indexed."""

    @cached_property
    def session(self) -> Session:
        """The lake's session, held here so it lives as long as this keeper does."""
        return self.lake.session()

    @property
    def cache(self) -> Path:
        """The read-through cache materialized lake-only files land in, mirroring their paths."""
        return self.lake.out / _CACHE

    @property
    def root(self) -> Path:
        return self.lake.root

    def __init__(self, lake: Lake) -> None:
        self.lake = lake
        self.blobs = Blobs(lake)

    def _audited(
        self, connection: duckdb.DuckDBPyConnection, sizes: Mapping[str, int]
    ) -> dict[str, Audit]:
        """The audit of each object named in `sizes`, its expected bytes, a window at a time."""
        audits: dict[str, Audit] = {}
        for window in windows(sorted(sizes), sizes.__getitem__, WINDOW_BYTES):
            audits |= self.blobs.audit(connection, window)
        return audits

    def _commit(self, connection: duckdb.DuckDBPyConnection, group: Sequence[_Entry]) -> None:
        """One window's new objects and index rows, inside the caller's transaction."""
        self.blobs.stage(
            connection,
            {
                entry.sha256: entry.source if entry.payload is None else entry.payload
                for entry in group
                if entry.new
            },
        )
        stamp = datetime.now(UTC)
        insert(
            connection,
            schema.evidence_log,
            [
                {
                    "ts": stamp,
                    "path": entry.path,
                    "sha256": entry.sha256,
                    "size": entry.size,
                    "media_type": entry.media_type,
                    **_labels(entry.path),
                }
                for entry in group
            ],
        )

    def _files(self, paths: Sequence[Path], *, root: Path | None = None) -> list[Path]:
        """Every file at or under `paths`, absolute and once each.

        Raises MissionError for what lies outside the workspace (or `root`) or behind a link.
        """
        if not paths:
            raise MissionError("name the evidence files or directories to keep")
        found: set[Path] = set()
        for given in paths:
            path = Path(os.path.abspath(given))
            within(root or self.root, path=path)
            found.update(tree_of(path, root=root or self.root))
        return sorted(found)

    def _read(self, source: Path, *, base: Path) -> _Entry:
        """Read `source` once, through its folders under `base` and without following a link.

        Its digest, what it holds and, unless it is larger than `_HELD_BYTES`, its bytes; a larger
        file is hashed as a stream and streamed again when staged. Indexed at its place under
        `base`. Raises MissionError when the file changed while it was read.
        """
        with OwnedFile(source, root=base).read() as stream:
            before = identity_of(os.fstat(stream.fileno()))
            size = before[2]
            payload = stream.read() if size <= _HELD_BYTES else None
            if payload is None:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
                stream.seek(0)
                head = stream.read(len(_MAGIC))
            else:
                digest, head = hashlib.sha256(payload).hexdigest(), payload[: len(_MAGIC)]
            if identity_of(os.fstat(stream.fileno())) != before:
                raise MissionError(f"{source} changed while reading; not ingested")
        return _Entry(
            source=source,
            path=source.relative_to(base).as_posix(),
            sha256=digest,
            size=size,
            media_type=_media_type(source, head),
            payload=payload,
        )

    def adopt(self, named: Mapping[str, str]) -> int:
        """Index each path at the object with its digest this lake keeps, moving no bytes.

        Answers how many paths gained a row. A host leaves unsent every file whose bytes the
        lake keeps, so its path is all that crosses. Size and media type are those the object
        was kept with, the type named by the new path's suffix first, as `ingest` names it.

        Raises MissionError when the lake keeps no intact object for one of them.
        """
        if not named:
            return 0
        index = schema.evidence
        listed = self.session.rows(
            f"SELECT DISTINCT ON (sha256) sha256, size, media_type FROM {ALIAS}"
            ".evidence WHERE sha256 IN (SELECT unnest(?::VARCHAR[]))",
            [sorted(set(named.values()))],
        )
        audits = self.session.run(
            partial(self._audited, sizes={digest: size for digest, size, _ in listed})
        )
        known = {
            digest: (size, media)
            for digest, size, media in listed
            if digest in audits and audits[digest].intact
        }
        if lost := sorted(path for path, digest in named.items() if digest not in known):
            raise MissionError(f"the lake keeps no object {named[lost[0]][:12]} for {lost[0]}")
        current = dict(self.session.rows(select(index.c.path, index.c.sha256)))
        stamp = datetime.now(UTC)
        rows = [
            {
                "ts": stamp,
                "path": path,
                "sha256": digest,
                "size": known[digest][0],
                "media_type": _media_type(
                    Path(path), _MAGIC if known[digest][1] == _PARQUET else b""
                ),
                **_labels(path),
            }
            for path, digest in sorted(named.items())
            if current.get(path) != digest
        ]
        return self.lake.append(schema.evidence_log, rows) if rows else 0

    def finalize(self, paths: Sequence[Path]) -> Finalized:
        """Record every file at or under `paths` as its producer closed it.

        The producer promises that nothing writes these files any more, which is what lets
        `evict` remove them: nothing here can tell a writer that still has one open.

        Raises MissionError for a path missing, outside the workspace or behind a link.
        """
        files: dict[Path, Identity] = {}
        for source in self._files(paths):
            with OwnedFile(source, root=self.root).read() as stream:
                files[source] = identity_of(os.fstat(stream.fileno()))
        return Finalized(files=files)

    def indexed(self, prefix: str = "") -> list[Located]:
        """Every indexed file at or under the workspace-relative `prefix`, all when empty."""
        if not self.lake.exists():
            return []
        return located(self.session.rows(listing(prefix)))

    def ingest(self, paths: Iterable[Path], *, staged: Path | None = None) -> Kept:
        """Keep every file at or under `paths` in the lake, byte for byte, and index its path.

        Idempotent: an intact object is not rewritten, and a path whose latest row already
        names its digest gets no new row. Files are taken in windows of at most `STAGED_BYTES`,
        each file read once by a pool of readers (opening a file, not the lake, is what costs on
        a scanned disk), and each window commits as one insert, one data file. An
        interrupted ingest keeps the windows it committed and a rerun finishes the rest. The run
        is recorded in `imports`.

        staged: a directory standing in for the workspace root, its files indexed where they
            will stand in the tree; a collected transfer is kept from there, never entering it.

        Raises MissionError for a path that is missing or outside this workspace.
        """
        given = [Path(os.path.abspath(path)) for path in paths]
        base = staged or self.root
        sources = self._files(given, root=base)
        self.lake.ready()
        index = schema.evidence
        current = dict(self.session.rows(select(index.c.path, index.c.sha256)))
        indexed = objects = size = 0
        with ThreadPoolExecutor(max_workers=READERS) as pool:
            # Each window of files is read, then committed as one insert, so memory stays
            # bounded and an interrupted ingest never leaves half an object.
            for window in windows(
                sources, lambda source: source.stat().st_size, blobs.Blobs.STAGED_BYTES
            ):
                batch = list(pool.map(partial(self._read, base=base), window))
                held = self.session.run(
                    partial(self.blobs.held, digests={entry.sha256 for entry in batch})
                )
                entries: list[_Entry] = []
                for entry in batch:
                    new = entry.sha256 not in held
                    held.add(entry.sha256)
                    if new or current.get(entry.path) != entry.sha256:
                        entries.append(entry.model_copy(update={"new": new}))
                if entries:
                    self.lake.transact(partial(self._commit, group=entries))
                indexed += len(entries)
                objects += sum(entry.new for entry in entries)
                size += sum(entry.size for entry in entries if entry.new)
        kept = Kept(files=len(sources), indexed=indexed, objects=objects, size=size)
        self.lake.append(
            schema.imports,
            [
                {
                    "ts": datetime.now(UTC),
                    "source": " ".join(posix(path.relative_to(base)) for path in given),
                    "destination": "evidence_log",
                    "rows": kept.indexed,
                }
            ],
        )
        return kept
