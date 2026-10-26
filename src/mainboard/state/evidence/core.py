"""Reading evidence back: recalled, materialized, inventoried, evicted, verified, replicated."""

import logging
from collections.abc import Collection, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from functools import partial
from pathlib import Path, PurePosixPath
from typing import Literal

import duckdb

from ...core.errors import MissionError
from ..constants import READERS, WINDOW_BYTES
from ..lake import Finding
from ..storage.blobs import Audit
from ..storage.owned import OwnedFile
from .keeping import Keeping, identity_of, listing, windows, within
from .models import Finalized, Identity, Inventory, Located, is_archived, located
from .replica import DirectoryReplica

logger = logging.getLogger(__name__)


def _standing(root: Path, row: Located) -> str | None:
    """The digest of the file at `row`'s path under `root`, None when no file is there.

    Raises MissionError for a link on the way, or a file that is not a regular one.
    """
    with suppress(FileNotFoundError):
        return OwnedFile(root / row.path, root=root).digest()
    return None


class Evidence(Keeping):
    """One workspace's evidence in its lake: kept, indexed, read back and verified."""

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

    def _copy(
        self, replica: DirectoryReplica, digest: str, connection: duckdb.DuckDBPyConnection
    ) -> None:
        """Put the object `digest` into `replica`, a chunk at a time from this lake."""
        replica.put(digest, self.blobs.chunks(connection, digest))

    def _evict(self, item: Inventory, closed: Identity) -> bool:
        """Remove `item`'s tree copy if it is still the closed file, False if it is not."""
        try:
            return self._vacated(item, closed)
        except FileNotFoundError:
            return False

    def _indexed_audits(self) -> tuple[list[Located], dict[str, Audit]]:
        """Every indexed file with the audits of the objects, on a fresh read-only attach."""
        with self.lake.open() as connection:
            rows = located(self.lake.execute(connection, listing()).fetchall())
            return rows, self._audited(connection, {row.sha256: row.size for row in rows})

    def _observed(self, row: Located) -> tuple[str, Literal["missing", "identical", "changed"]]:
        """Digest the tree copy of `row` and compare it; a link or a device there has no digest."""
        try:
            digest = _standing(self.root, row)
        except MissionError:
            return "", "changed"
        if digest is None:
            return "", "missing"
        return digest, "identical" if digest == row.sha256 else "changed"

    def _vacated(self, item: Inventory, closed: Identity) -> bool:
        owned = OwnedFile(self.root / item.path, root=self.root)
        with owned.quarantine() as isolated:
            if (
                identity_of(owned.stat_at(isolated)) != closed
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
            if identity_of(owned.stat_at(isolated)) != closed:
                logger.warning("%s changed while the lake was checked; kept", item.path)
                return False
            owned.unlink_at(isolated)
        return True

    def _write(self, row: Located, *, root: Path, replace: bool = False) -> None:
        """Write `row`'s object to its path under `root`, refusing when the lake lost it."""
        if not self.session.run(
            lambda connection: self.blobs.write(
                connection, row.sha256, root / row.path, root=root, size=row.size, replace=replace
            )
        ):
            raise MissionError(f"the lake holds no intact copy of {row.path} ({row.sha256[:12]})")

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
            if not is_archived(within(self.root, path=source))
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

    def inventory(self, paths: Sequence[Path]) -> list[Inventory]:
        """Where every indexed file at or under `paths` stands, its tree copy against the lake.

        A read and nothing more: no byte of the tree and no row or checksum of the lake changes,
        and a workspace without a lake answers empty. The lake's side is the blob verifier's own,
        so what reads intact here is what `evict` and every reader accept.
        """
        wanted = {within(self.root, path=path) for path in paths}
        if not self.lake.exists():
            return []
        with self.lake.open() as connection:
            listed = located(self.lake.execute(connection, listing()).fetchall())
            rows = [row for row in listed if self._is_under(row.path, wanted)]
            audits = self._audited(connection, {row.sha256: row.size for row in rows})
        with ThreadPoolExecutor(max_workers=READERS) as pool:
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
            prefix = within(self.root, path=given)
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

    def recall(self, sha256: str) -> bytes | None:
        """The object with `sha256`, verified, None when this lake holds no intact copy."""
        if not self.lake.exists():
            return None
        found = self.session.run(lambda connection: self.blobs.read(connection, {sha256}))
        return found.get(sha256)

    def recalled(self, sizes: Mapping[str, int]) -> Iterator[tuple[str, bytes]]:
        """Each object among `sizes` this lake holds intact, verified and read in windows.

        `sizes` maps digest to size, and a window holds at most `WINDOW_BYTES`, one scan of
        `blobs` each. A digest says nothing about which data file holds it, so `recall` scans every
        file's digests per object: settling a receipt that pins hundreds of artifacts outlived the
        settler's deadline on every pass (2026-10-22).
        """
        if not self.lake.exists():
            return
        for window in windows(sorted(sizes), sizes.__getitem__, WINDOW_BYTES):
            yield from self.session.run(partial(self.blobs.read, digests=window)).items()

    def replicate(self, replica: DirectoryReplica) -> int:
        """Copy every indexed object `replica` lacks into it, then the whole index.

        It answers how many objects were copied.

        The missing objects are read in windows of at most `WINDOW_BYTES`, one scan of `blobs`
        each, since a query per object pays its planning every time (nine objects a second
        across 330,000). An object larger than a window, or one a window could not read whole,
        is copied on its own a chunk at a time.
        """
        rows = self.indexed()
        sizes = {row.sha256: row.size for row in rows}
        missing = sorted(set(sizes) - replica.holds(sizes))
        alone = [digest for digest in missing if sizes[digest] > WINDOW_BYTES]
        windowed = [digest for digest in missing if sizes[digest] <= WINDOW_BYTES]
        for window in windows(windowed, sizes.__getitem__, WINDOW_BYTES):
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
