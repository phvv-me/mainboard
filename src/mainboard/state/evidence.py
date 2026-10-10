# EXPERIMENT EVIDENCE KEPT IN THE LAKE, so its files can leave version control and every reader
# still finds their bytes.
#
# An evidence file's bytes are one content-addressed object in `blobs` (chunked, see
# `state.blobs`), and `evidence_log` records which workspace-relative path held which object; the
# `evidence` view is each path's latest record. Bytes are kept exactly as written, never decoded
# or re-encoded, so every SHA-256 a `receipts.json` or an artifact reference pins keeps verifying
# whether the bytes come from disk or from the lake. An ingest commits in groups of whole files,
# each group one insert, so small files share Parquet data files instead of one each.
#
# INTEGRITY. `verify` puts the object of every indexed path through the one blob verifier
# (`state.blobs`), a window of objects at a time, and reports each path whose object is missing,
# short or damaged. It never writes. A chunk with no checksum gets none recorded, and `ingest`
# repairs.
#
# CUSTODY. A tree copy leaves only on proof. `inventory` reads where every indexed file stands
# against the lake and changes nothing, and `evict` consumes that same inventory for the files
# their producer has `finalize`d: each is moved by name into a private folder beside it, hashed
# and checked against the lake once more, and only then removed, so a file replaced or written
# meanwhile is put back and never unlinked. Without that handoff nothing is evicted, since
# renaming fences no writer that still has the file open. The parts of an artifact archive are
# never evicted: readers open its members by path. `materialize` goes the other way as carefully,
# through `OwnedFile`, and refuses a changed file already there, naming both digests.
#
# READERS. What used to glob a directory asks an `EvidenceTree` instead: the files on disk, and
# every file the lakes of the workspaces holding that directory index under it, materialized on
# first use into the lake's read-through cache (`<state>/evidence/<path>`), outside the tree and
# outside version control. A read pinned by digest needs no path at all: the object is found by
# its SHA-256 in any of those lakes. A nested workspace (a research repository with its own
# manifest inside the monorepo) is served by its own lake first and the enclosing one after.
#
# REPLICATION. Once evidence leaves version control the lake is its only copy, so a second one
# on another disk is what makes that safe. `Replica` is the shape any home implements and
# `Evidence.replicate` the one call that fills it: every object the replica lacks, then the whole
# path index, so both the bytes and which path held them survive the lake's disk.
# `DirectoryReplica` keeps them on a mounted disk (the center's external drive).

import hashlib
import logging
import mimetypes
import os
from abc import ABC, abstractmethod
from collections.abc import Callable, Collection, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from datetime import UTC, datetime
from functools import cache, cached_property, partial
from pathlib import Path, PurePosixPath
from typing import Literal

import duckdb
from patos import FrozenModel
from sqlalchemy import Select, func, or_, select

from ..core.errors import MissionError
from ..core.project import Project
from . import schema
from .blobs import STAGED_BYTES, Audit, Blobs
from .lake import Finding, Lake, Session, insert
from .owned import OwnedFile, link_in, tree_of

# Where the read-through cache lives under the state directory.
CACHE = "evidence"

# A file as the filesystem knows it: device, inode, size and modification time in nanoseconds.
type Identity = tuple[int, int, int, int]

logger = logging.getLogger(__name__)

# The folder `trials.archive.ParquetArtifacts` packs artifacts into. Readers address their members
# by path, opening its parts in the tree and never recalling them by digest, so the parts stay.
_ARCHIVE = "artifacts.parquet"

# How many files are read at once: opening a file is what costs on a scanned disk, and
# the opens overlap. The largest file a read holds in memory; a larger one streams when staged.
_READERS = 16
_HELD_BYTES = 32 << 20

# The most object bytes one replicating read gathers: DuckDB holds a read's whole answer (half
# the center lake in one read peaked at 32 GB), and a window costs one scan of `blobs`.
_WINDOW_BYTES = 1 << 30

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


class Kept(FrozenModel):
    """What one `ingest` did.

    files: evidence files read.
    indexed: paths given a new index row, their content new or changed.
    objects: objects new to the lake, and `size` how many bytes those were.
    """

    files: int = 0
    indexed: int = 0
    objects: int = 0
    size: int = 0


class Located(FrozenModel):
    """One path's latest index row: which object it holds."""

    path: str
    sha256: str
    size: int


class Inventory(FrozenModel):
    """Where one indexed file stands, its tree copy against the lake's, as observed once.

    sha256: the digest the lake indexes at the path, and `size` its bytes.
    local_sha256: the digest of the file now at the path, empty where there is none to read:
        no file at all, a link, or anything but a regular file.
    local: whether the tree holds no file, the same bytes or other ones.
    lake: whether the lake holds no row of the object, an intact object or a damaged one.
    """

    path: str
    sha256: str
    size: int
    local_sha256: str
    local: Literal["missing", "identical", "changed"]
    lake: Literal["missing", "intact", "corrupt"]

    @property
    def archived(self) -> bool:
        """Whether the file is a part of an artifact archive, which readers open by path."""
        return _is_archived(self.path)

    @property
    def evictable(self) -> bool:
        """Whether the tree copy is the lake's whole object and nothing reads it by path."""
        return self.local == "identical" and self.lake == "intact" and not self.archived


class Finalized(FrozenModel):
    """Files their producer closed, as they stood when it said so, the only ones to evict.

    files: each file with its identity at that moment. The producer promises that no writer
        remains. A file that no longer matches has broken the promise, so its bytes stay.
    """

    files: dict[Path, Identity]


class Replica(ABC):
    """A second durable copy of the lake's evidence objects, keyed by SHA-256, and of the index
    saying which path held which object."""

    @abstractmethod
    def holds(self, digests: Collection[str]) -> set[str]:
        """The digests among `digests` this replica already keeps intact."""

    @abstractmethod
    def put(self, digest: str, chunks: Iterator[bytes]) -> None:
        """Keep `digest`'s object, handed over as its ordered chunks; verify before keeping."""

    @abstractmethod
    def index(self, rows: Sequence[Located]) -> None:
        """Replace the replica's path index with `rows`, the lake's whole current index."""


class DirectoryReplica(Replica):
    """A replica on a mounted disk: each object one file at `<root>/<first two hex>/<sha256>`,
    written beside its place and renamed in only once its digest checks, so an interrupted copy
    never leaves wrong bytes under a good name; the index is `<root>/index.jsonl`."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def holds(self, digests: Collection[str]) -> set[str]:
        return {digest for digest in digests if self._object(digest).is_file()}

    def put(self, digest: str, chunks: Iterator[bytes]) -> None:
        target = self._object(digest)
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_suffix(".partial")
        hasher = hashlib.sha256()
        with partial.open("wb") as sink:
            for chunk in chunks:
                hasher.update(chunk)
                sink.write(chunk)
        if hasher.hexdigest() != digest:
            partial.unlink()
            raise MissionError(
                f"object {digest} read back as {hasher.hexdigest()}; not replicated"
            )
        partial.replace(target)

    def index(self, rows: Sequence[Located]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        partial = self.root / "index.jsonl.partial"
        partial.write_text(
            "".join(f"{row.model_dump_json()}\n" for row in rows), encoding="utf-8", newline="\n"
        )
        partial.replace(self.root / "index.jsonl")

    def _object(self, digest: str) -> Path:
        return self.root / digest[:2] / digest


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


class Evidence:
    """One workspace's evidence in its lake: kept, indexed, read back and verified."""

    def __init__(self, lake: Lake) -> None:
        self.lake = lake
        self.blobs = Blobs(lake)

    @property
    def cache(self) -> Path:
        """The read-through cache materialized lake-only files land in, mirroring their paths."""
        return self.lake.out / CACHE

    @property
    def root(self) -> Path:
        return self.lake.root

    @cached_property
    def session(self) -> Session:
        """The lake's session, held here so it lives as long as this keeper does."""
        return self.lake.session()

    def cached(self, row: Located) -> Path:
        """`row`'s file in the read-through cache, materialized first when it is not there.

        The cache is the lake's own, so a file gone stale at the path is replaced.
        """
        target = self.cache / row.path
        try:
            current = OwnedFile(target, root=self.cache).digest() == row.sha256
        except FileNotFoundError:
            current = False
        if not current:
            self._write(row, root=self.cache, replace=True)
        return target

    def indexed(self, prefix: str = "") -> list[Located]:
        """Every indexed file at or under the workspace-relative `prefix`, all when empty."""
        if not self.lake.exists():
            return []
        return _located(self.session.rows(_listing(prefix)))

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
        with ThreadPoolExecutor(max_workers=_READERS) as pool:
            # Each window of files is read, then committed as one insert, so memory stays
            # bounded and an interrupted ingest never leaves half an object.
            for window in _windows(sources, lambda source: source.stat().st_size, STAGED_BYTES):
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
                    "source": " ".join(_posix(path.relative_to(base)) for path in given),
                    "destination": "evidence_log",
                    "rows": kept.indexed,
                }
            ],
        )
        return kept

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
            f"SELECT DISTINCT ON (sha256) sha256, size, media_type FROM {schema.ALIAS}"
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

    def materialize(self, paths: Sequence[Path], into: Path | None = None) -> list[Path]:
        """Write the missing indexed files at or under `paths` back, returning those written.

        They go under `into`, the workspace by default. A file already there with the lake's bytes
        is left alone. One with other bytes is never touched, and nothing is written while any
        such conflict stands.

        Raises MissionError when a path indexes nothing, the lake lost an object, a link stands in
        the way, or a changed file does, naming each such path with both digests.
        """
        root = into or self.root
        wanted: dict[str, Located] = {}
        for given in paths:
            prefix = within(self.root, given)
            if not (rows := self.indexed(prefix)):
                raise MissionError(f"the lake indexes no evidence at or under {prefix}")
            wanted.update((row.path, row) for row in rows)
        standing = [(row, _standing(root, row)) for row in wanted.values()]
        if conflicts := [
            f"{root / row.path}: destination {found}; lake {row.sha256}"
            for row, found in standing
            if found not in {None, row.sha256}
        ]:
            raise MissionError("refusing to overwrite changed files\n" + "\n".join(conflicts))
        missing = [row for row, found in standing if found is None]
        for row in missing:
            self._write(row, root=root)
        return [root / row.path for row in missing]

    def finalize(self, paths: Sequence[Path]) -> Finalized:
        """Record every file at or under `paths` as its producer closed it.

        The producer promises that nothing writes these files any more, which is what lets
        `evict` remove them: nothing here can tell a writer that still has one open.

        Raises MissionError for a path missing, outside the workspace or behind a link.
        """
        files: dict[Path, Identity] = {}
        for source in self._files(paths):
            with OwnedFile(source, root=self.root).read() as stream:
                files[source] = _identity(os.fstat(stream.fileno()))
        return Finalized(files=files)

    def inventory(self, paths: Sequence[Path]) -> list[Inventory]:
        """Where every indexed file at or under `paths` stands, its tree copy against the lake.

        A read and nothing more: no byte of the tree and no row or checksum of the lake changes,
        and a workspace without a lake answers empty. The lake's side is the blob verifier's own,
        so what reads intact here is what `evict` and every reader accept.
        """
        wanted = {within(self.root, path) for path in paths}
        if not self.lake.exists():
            return []
        with self.lake.open() as connection:
            listed = _located(self.lake.execute(connection, _listing()).fetchall())
            rows = [row for row in listed if self._is_under(row.path, wanted)]
            audits = self._audited(connection, {row.sha256: row.size for row in rows})
        with ThreadPoolExecutor(max_workers=_READERS) as pool:
            observed = pool.map(self._observed, rows)
        return [
            Inventory(
                path=row.path,
                sha256=row.sha256,
                size=row.size,
                local_sha256=digest,
                local=local,
                lake=self._condition(audits.get(row.sha256), row.size),
            )
            for row, (digest, local) in zip(rows, observed, strict=True)
        ]

    def evict(self, finalized: Finalized) -> list[Path]:
        """Remove the tree copy of each finalized file the lake holds whole, returning them.

        Any other file stays. Each candidate is moved by name into a private folder beside it,
        hashed again, matched with its identity when the producer closed it and with the lake once
        more, and only then removed, so a file that was replaced or written meanwhile is put back
        and never lost. Readers recall an evicted file by digest. `materialize` writes it back.
        Artifact archive parts always stay, since their members are read by path.

        Raises MissionError when a retained file cannot go back because a new one took its name.
        """
        closed = {
            source: identity
            for source, identity in finalized.files.items()
            if not _is_archived(within(self.root, source))
        }
        if parts := len(finalized.files) - len(closed):
            logger.warning("%d archive files stay: readers open them by path", parts)
        evicted: list[Path] = []
        for item in self.inventory(list(closed)):
            source = self.root / item.path
            if item.evictable and self._evict(item, closed[source]):
                evicted.append(source)
        for source in evicted:
            # Folders the file leaves empty go too; the workspace root never is, holding its
            # manifest.
            for folder in source.relative_to(self.root).parents[:-1]:
                with suppress(OSError):
                    (self.root / folder).rmdir()
        return evicted

    def _evict(self, item: Inventory, closed: Identity) -> bool:
        """Remove `item`'s tree copy if it is still the closed file, False if it is not."""
        try:
            return self._vacated(item, closed)
        except FileNotFoundError:
            return False

    def _vacated(self, item: Inventory, closed: Identity) -> bool:
        owned = OwnedFile(self.root / item.path, root=self.root)
        with owned.quarantine() as isolated:
            if (
                _identity(owned.stat_at(isolated)) != closed
                or owned.digest_at(isolated) != item.sha256
            ):
                logger.warning("%s changed since its producer closed it; kept", item.path)
                return False
            audit = self.session.run(
                lambda connection: self.blobs.inspect(connection, item.sha256)
            )
            if fault := audit.fault(item.size):
                logger.warning("%s kept: the lake's copy has %s", item.path, fault)
                return False
            if _identity(owned.stat_at(isolated)) != closed:
                logger.warning("%s changed while the lake was checked; kept", item.path)
                return False
            owned.unlink_at(isolated)
        return True

    def _observed(self, row: Located) -> tuple[str, Literal["missing", "identical", "changed"]]:
        """Digest the tree copy of `row` and compare it; a link or a device there has no digest."""
        try:
            digest = _standing(self.root, row)
        except MissionError:
            return "", "changed"
        if digest is None:
            return "", "missing"
        return digest, "identical" if digest == row.sha256 else "changed"

    @staticmethod
    def _condition(audit: Audit | None, size: int) -> Literal["missing", "intact", "corrupt"]:
        """How the lake holds an object of `size` bytes, from its audit."""
        if audit is None:
            return "missing"
        return "corrupt" if audit.fault(size) else "intact"

    @staticmethod
    def _is_under(path: str, prefixes: Collection[str]) -> bool:
        """Whether the index `path` is one of `prefixes` or lies under one; "" is the root."""
        return (
            "" in prefixes
            or path in prefixes
            or any(parent.as_posix() in prefixes for parent in PurePosixPath(path).parents)
        )

    def recall(self, sha256: str) -> bytes | None:
        """The object with `sha256`, verified, None when this lake holds no intact copy."""
        if not self.lake.exists():
            return None
        found = self.session.run(lambda connection: self.blobs.read(connection, {sha256}))
        return found.get(sha256)

    def replicate(self, replica: Replica) -> int:
        """Copy every indexed object `replica` lacks into it, then the whole index; how many
        objects were copied.

        The missing objects are read in windows of at most `_WINDOW_BYTES`, one scan of `blobs`
        each, since a query per object pays its planning every time (nine objects a second
        across 330,000). An object larger than a window, or one a window could not read whole,
        is copied on its own a chunk at a time.
        """
        rows = self.indexed()
        sizes = {row.sha256: row.size for row in rows}
        missing = sorted(set(sizes) - replica.holds(sizes))
        alone = [digest for digest in missing if sizes[digest] > _WINDOW_BYTES]
        windowed = [digest for digest in missing if sizes[digest] <= _WINDOW_BYTES]
        for window in _windows(windowed, sizes.__getitem__, _WINDOW_BYTES):
            found = self.session.run(partial(self.blobs.read, digests=window))
            for digest, whole in found.items():
                replica.put(digest, iter((whole,)))
            alone += [digest for digest in window if digest not in found]
        for digest in alone:
            self.session.run(partial(self._copy, replica, digest))
        replica.index(rows)
        return len(missing)

    def verify(self) -> list[Finding]:
        """Find every indexed path whose object is missing, short or damaged.

        One finding per path whose object is absent, lacks a chunk, holds the wrong number of
        bytes or fails a recorded checksum or its SHA-256. Every indexed object is read through
        the blob verifier a window at a time, and nothing is written: a chunk kept before
        checksums existed is checked by its SHA-256 and gets no checksum recorded. Chunks DuckDB
        cannot read back at all (a data file damaged past decoding) are one finding.
        """
        if not self.lake.exists():
            return []
        try:
            rows, audits = self._indexed_audits()
        except duckdb.Error as fault:
            return [Finding(table="blobs", kind="unreadable", detail=str(fault).splitlines()[0])]
        faults = {
            row: audits[row.sha256].fault(row.size) if row.sha256 in audits else "no copy held"
            for row in rows
        }
        return [
            Finding(
                table="evidence_log",
                kind="evidence",
                detail=f"{row.path} ({row.sha256[:12]}): {fault}",
            )
            for row, fault in faults.items()
            if fault
        ]

    def _indexed_audits(self) -> tuple[list[Located], dict[str, Audit]]:
        """Every indexed file with the audits of the objects, on a fresh read-only attach."""
        with self.lake.open() as connection:
            rows = _located(self.lake.execute(connection, _listing()).fetchall())
            return rows, self._audited(connection, {row.sha256: row.size for row in rows})

    def _audited(
        self, connection: duckdb.DuckDBPyConnection, sizes: Mapping[str, int]
    ) -> dict[str, Audit]:
        """The audit of each object named in `sizes`, its expected bytes, a window at a time."""
        audits: dict[str, Audit] = {}
        for window in _windows(sorted(sizes), sizes.__getitem__, _WINDOW_BYTES):
            audits |= self.blobs.audit(connection, window)
        return audits

    def _read(self, source: Path, *, base: Path) -> _Entry:
        """Read `source` once, through its folders under `base` and without following a link.

        Its digest, what it holds and, unless it is larger than `_HELD_BYTES`, its bytes; a larger
        file is hashed as a stream and streamed again when staged. Indexed at its place under
        `base`. Raises MissionError when the file changed while it was read.
        """
        with OwnedFile(source, root=base).read() as stream:
            before = _identity(os.fstat(stream.fileno()))
            size = before[2]
            payload = stream.read() if size <= _HELD_BYTES else None
            if payload is None:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
                stream.seek(0)
                head = stream.read(len(_MAGIC))
            else:
                digest, head = hashlib.sha256(payload).hexdigest(), payload[: len(_MAGIC)]
            if _identity(os.fstat(stream.fileno())) != before:
                raise MissionError(f"{source} changed while reading; not ingested")
        return _Entry(
            source=source,
            path=source.relative_to(base).as_posix(),
            sha256=digest,
            size=size,
            media_type=_media_type(source, head),
            payload=payload,
        )

    def _copy(self, replica: Replica, digest: str, connection: duckdb.DuckDBPyConnection) -> None:
        """Put the object `digest` into `replica`, a chunk at a time from this lake."""
        replica.put(digest, self.blobs.chunks(connection, digest))

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
            within(root or self.root, path)
            found.update(tree_of(path, root=root or self.root))
        return sorted(found)

    def _write(self, row: Located, *, root: Path, replace: bool = False) -> None:
        """Write `row`'s object to its path under `root`, refusing when the lake lost it."""
        if not self.session.run(
            lambda connection: self.blobs.write(
                connection, row.sha256, root / row.path, root=root, size=row.size, replace=replace
            )
        ):
            raise MissionError(f"the lake holds no intact copy of {row.path} ({row.sha256[:12]})")


class EvidenceTree:
    """The evidence visible from one directory: its files on disk, and every file the lakes of
    the workspaces holding it index there, whether or not the file is still on disk."""

    def __init__(self, base: Path) -> None:
        self.base = Path(os.path.abspath(base))

    @property
    def keepers(self) -> list[Evidence]:
        """The evidence of every workspace holding `base` that keeps a lake, nearest first."""
        found: dict[str, Evidence] = {}
        project = Project()
        for directory in (self.base, *self.base.parents):
            if project.manifest(directory).is_file():
                evidence = _kept(directory)
                if evidence.lake.exists():
                    found.setdefault(evidence.lake.served or str(evidence.lake.catalog), evidence)
        return list(found.values())

    def keep(self, paths: Sequence[Path]) -> Kept | None:
        """Move `paths` into the center's lake, the nearest one holding `base`.

        They are kept byte for byte, then their tree copies dropped, since the lake is the
        evidence's home and the tree only its cache. The caller is their producer and has closed
        them (a finished run's receipts), which is what lets the copies go; one that was written
        since is kept in the tree. A path behind a link is left where it is, in the tree. None when
        the nearest lake is no center's: a dispatch mirror's staging lake is not a home, and the
        center collects the mirror's tree.
        """
        keepers = self.keepers[:1]
        if not (paths and keepers and keepers[0].lake.central):
            return None
        keeper = keepers[0]
        owned = [path for path in paths if link_in(path, root=keeper.root) is None]
        if len(owned) < len(paths):
            logger.warning(
                "%d of %d paths lie behind a link and stay", len(paths) - len(owned), len(paths)
            )
        if not owned:
            return None
        closed = keeper.finalize(owned)
        kept = keeper.ingest(owned)
        keeper.evict(closed)
        return kept

    def holds(self, digests: Collection[str]) -> set[str]:
        """The digests among `digests` some lake holding `base` keeps an object for."""
        return {
            digest
            for evidence in self.keepers
            for digest in evidence.session.run(partial(evidence.blobs.held, digests=digests))
        }

    def restore(self, paths: Iterable[Path]) -> list[Path]:
        """Write back every file at or under `paths` that the tree lacks and a lake holding
        `base` keeps, nearest lake first; the files written. A path no lake indexes is left
        missing, for its reader to fail on."""
        written: list[Path] = []
        for path in paths:
            for keeper in self.keepers:
                try:
                    written += keeper.materialize([path])
                except MissionError:
                    continue
                break
        return written

    def directories(self, pattern: str) -> list[Path]:
        """Every directory under `base` matching the glob `pattern`, in path order: on disk, or
        holding a file a lake indexes. Named where it would stand in the tree."""
        found = {
            path.relative_to(self.base).as_posix()
            for path in self.base.glob(pattern)
            if path.is_dir()
        }
        held = {
            parent
            for _, relative, _ in self._indexed()
            for parent in PurePosixPath(relative).parents
            if parent.parts
        }
        found |= {parent.as_posix() for parent in held if parent.full_match(pattern)}
        return [self.base / relative for relative in sorted(found)]

    def files(self, pattern: str) -> list[Path]:
        """Every file under `base` matching the glob `pattern`, in path order: from disk, else
        from the read-through cache of the lake indexing it."""
        found = {path.relative_to(self.base).as_posix(): path for path in self.base.glob(pattern)}
        for evidence, relative, row in self._indexed():
            if relative not in found and PurePosixPath(relative).full_match(pattern):
                found[relative] = evidence.cached(row)
        return [found[relative] for relative in sorted(found)]

    def read(self, path: Path, sha256: str) -> bytes:
        """The bytes pinned at `path` by `sha256`: from disk, else from a lake by digest.

        Raises FileNotFoundError when neither holds them, ValueError when the bytes found differ.
        """
        try:
            data = path.read_bytes()
        except FileNotFoundError:
            data = self.recall(sha256)
        if hashlib.sha256(data).hexdigest() != sha256:
            raise ValueError(f"{path} does not match its pinned digest")
        return data

    def recall(self, sha256: str) -> bytes:
        """The object with `sha256` from the first lake holding it intact.

        Raises FileNotFoundError when none does.
        """
        for evidence in self.keepers:
            data = evidence.recall(sha256)
            if data is not None:
                return data
        raise FileNotFoundError(f"no lake around {self.base} holds {sha256}")

    def _indexed(self) -> Iterator[tuple[Evidence, str, Located]]:
        """Each indexed file under `base`, with its keeper and its path relative to `base`."""
        for evidence in self.keepers:
            prefix = _posix(self.base.relative_to(evidence.root))
            for row in evidence.indexed(prefix):
                yield evidence, row.path.removeprefix(prefix).removeprefix("/"), row


@cache
def _kept(root: Path) -> Evidence:
    """One `Evidence` per workspace for the life of the process, so its lake session (an attach
    costs a tenth of a second) outlives the reader that asked, as a registry's does."""
    return Evidence(Lake.at(root))


def _posix(relative: Path) -> str:
    """A relative path as the index spells it, the workspace root itself as empty."""
    spelled = relative.as_posix()
    return "" if spelled == "." else spelled


def within(root: Path, path: Path) -> str:
    """`path` (absolute, or relative to the current directory) as the index spells it under the
    workspace `root`.

    Raises MissionError for a path outside the workspace.
    """
    absolute = Path(os.path.abspath(path))
    try:
        return _posix(absolute.relative_to(root))
    except ValueError:
        raise MissionError(f"{absolute} lies outside the workspace at {root}") from None


def _identity(metadata: os.stat_result) -> Identity:
    return metadata.st_dev, metadata.st_ino, metadata.st_size, metadata.st_mtime_ns


def _listing(prefix: str = "") -> Select:
    """Select every indexed file at or under the workspace-relative `prefix`, all when empty."""
    kept = schema.evidence
    query = select(kept.c.path, kept.c.sha256, kept.c.size).order_by(kept.c.path)
    under = or_(kept.c.path == prefix, func.starts_with(kept.c.path, f"{prefix}/"))
    return query.where(under) if prefix else query


def _is_archived(path: str) -> bool:
    """Whether the workspace-relative `path` is a file inside an artifact archive's folder."""
    return _ARCHIVE in PurePosixPath(path).parts[:-1]


def _located(rows: Iterable[tuple[str, str, int]]) -> list[Located]:
    return [Located(path=path, sha256=digest, size=size) for path, digest, size in rows]


def _standing(root: Path, row: Located) -> str | None:
    """The digest of the file at `row`'s path under `root`, None when no file is there.

    Raises MissionError for a link on the way, or a file that is not a regular one.
    """
    try:
        return OwnedFile(root / row.path, root=root).digest()
    except FileNotFoundError:
        return None


def _windows[T](items: Sequence[T], weight: Callable[[T], int], limit: int) -> Iterator[list[T]]:
    """`items` in order, in windows weighing at most `limit` together unless one item alone is
    heavier, so whatever a window holds at once stays bounded."""
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
    """What `path` holds, by its suffix, else by Parquet's leading magic in `head`, else raw
    bytes."""
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
