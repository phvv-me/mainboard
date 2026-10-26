"""The result exporter a host runs: `pack`, sent over ssh stdin as this file's text and run by
mainboard's own Python there, so it imports the standard library alone."""

import fnmatch
import hashlib
import json
import os
import sys
from collections.abc import Iterator, Mapping
from pathlib import Path, PurePosixPath
from stat import S_ISDIR, S_ISREG
from types import MappingProxyType
from zipfile import ZIP_ZSTANDARD, ZipFile

# The archive's last member when any file was left unsent because the center's lake keeps its
# bytes, as {path: digest}. Outside every collected scope, so no evidence file can take its name.
# Defined here, not in `constants`: this file is sent to the host as the script itself, and imports
# nothing but the standard library.
HELD = "mainboard-held.json"

# The zstd level every member is compressed at. Measured on 1.5 GB of cutok evidence on a GH200
# node (2026-10-09): level 1 sent 3.5x fewer bytes for 3.7 s of one core, level 3 3.9x for 7.5 s,
# level 6 4.3x for 16.6 s; at a few MB/s over ssh, level 3's bytes save more than its CPU costs.
_LEVEL = 3


def pack(
    root: str,
    *,
    relative: str,
    kept: Mapping[str, str] = MappingProxyType({}),
    known: Mapping[str, str] = MappingProxyType({}),
    cache: str = "",
) -> None:
    """Stream regular files under one workspace-relative path without following links.

    kept: the digest the center's lake indexes at each workspace-relative path. A file there with
        those bytes is not sent, and neither is any file holding a digest the lake keeps: it is
        named in `HELD`, and the center indexes its path from the object it already has.
    known: digests of files the center holds on disk outside its lake, by path, not sent again.
    cache: workspace-relative file of remote digests, so a file is hashed again only after its
        size or modification time changed.
    """
    base = Path(root).expanduser().resolve(strict=True)
    digests = Digests(base / cache if cache else None, base=base)
    known, held = {**kept, **known}, set(kept.values())
    with ZipFile(sys.stdout.buffer, "w", ZIP_ZSTANDARD, compresslevel=_LEVEL) as archive:
        writer = _Writer(archive, base=base, digests=digests, held=held, known=known)
        for path, stat in _paths(base, relative):
            writer.add(path, stat)
        if writer.named:
            archive.writestr(HELD, json.dumps(writer.named))
    digests.save()


class Digests:
    """SHA-256 of workspace files, reused while a file keeps its size and modification time."""

    def __init__(self, path: Path | None, *, base: Path) -> None:
        self.path = path
        self.base = base
        try:
            self.held = json.loads(path.read_text(encoding="utf-8")) if path else {}
        except OSError, ValueError:
            self.held = {}
        self.changed = False

    def of(self, path: Path, stat: os.stat_result) -> str:
        held = self.held.get(path.relative_to(self.base).as_posix())
        if held and held[:2] == [stat.st_size, stat.st_mtime_ns]:
            return held[2]
        return self.record(path, stat, sha256(path))

    def record(self, path: Path, stat: os.stat_result, digest: str) -> str:
        """Remember `digest` for `path` as it stands at `stat`, and answer it."""
        self.held[path.relative_to(self.base).as_posix()] = [
            stat.st_size,
            stat.st_mtime_ns,
            digest,
        ]
        self.changed = True
        return digest

    def save(self) -> None:
        """Write the cache atomically, so a concurrent pass reads the old one or the new one."""
        if not (self.path and self.changed):
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        staged = self.path.with_name(f"{self.path.name}.{os.getpid()}.tmp")
        staged.write_text(json.dumps(self.held), encoding="utf-8")
        os.replace(staged, self.path)


class _Writer:
    """Archives a pass's files, naming in `named` each one whose digest the lake keeps."""

    @staticmethod
    def _range(data: bytes) -> tuple[int, int]:
        """Validate contiguous byte offsets without reserializing event records."""
        start = json.loads(data.split(b"\n", 1)[0])["offset"]
        if type(start) is not int or start < 0:
            raise ValueError("invalid event offset")
        position = start
        for line in data.split(b"\n")[:-1]:
            if json.loads(line)["offset"] != position:
                raise ValueError("noncontiguous event offsets")
            position += len(line) + 1
        return start, position

    def __init__(
        self,
        archive: ZipFile,
        *,
        base: Path,
        digests: Digests,
        held: set[str],
        known: Mapping[str, str],
    ) -> None:
        self.archive = archive
        self.base = base
        self.digests = digests
        self.held = held
        self.known = known
        self.named: dict[str, str] = {}

    def _immutable(self, path: Path) -> None:
        """Copy one file and refuse an observed size or timestamp change."""
        before = path.stat()
        self.archive.write(path, path.relative_to(self.base).as_posix())
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError("file changed during collection: " + str(path))

    def _snapshot(self, path: Path) -> None:
        """Retain complete event records byte-for-byte under their validated offset range.

        A range the center already collected is not sent again.
        """
        with path.open("rb") as stream:
            data = stream.read(path.stat().st_size)
        data = data[: data.rfind(b"\n") + 1]
        if not data:
            return
        start, end = self._range(data)
        name = path.with_name(f"collected-{start:020d}-{end:020d}.ndjson").relative_to(self.base)
        if name.as_posix() not in self.known:
            self.archive.writestr(name.as_posix(), data)

    def add(self, path: Path, stat: os.stat_result) -> None:
        """Send `path` unless the center already holds its bytes."""
        if path.parent.name == "events" and path.name == "live.ndjson":
            self._snapshot(path)
            return
        key = path.relative_to(self.base).as_posix()
        digest = self.digests.of(path, stat)
        if self.known.get(key) == digest:
            return
        if digest in self.held:
            self.named[key] = digest
        else:
            self._immutable(path)


def sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _paths(base: Path, relative: str) -> Iterator[tuple[Path, os.stat_result]]:
    """Enumerate contained regular files with their one `lstat`, retaining traversal failures.

    The selection is checked to lie inside the workspace once; below it nothing is followed,
    so every regular file found lies inside too, and any link refuses the collection.
    """
    selected = base.joinpath(*PurePosixPath(relative).parts)
    try:
        resolved = selected.resolve(strict=True)
    except FileNotFoundError:
        # A job that declared a results path and never wrote it left nothing to collect: an empty
        # selection, not a failed transfer that every later pass would retry for ever.
        return
    if not resolved.is_relative_to(base):
        raise ValueError("collection path escapes the remote workspace")
    for path, stat in walked(selected):
        if not excluded(path):
            yield _regular(path, stat)


def _regular(path: Path, stat: os.stat_result) -> tuple[Path, os.stat_result]:
    """`path` with its status, refusing anything but a regular file."""
    if not S_ISREG(stat.st_mode):
        raise ValueError("collection contains a non-regular file: " + str(path))
    return path, stat


def walked(selected: Path) -> Iterator[tuple[Path, os.stat_result]]:
    """Walk a directory by `scandir`, one `lstat` per entry, a link yielded as itself."""
    stat = selected.lstat()
    if not S_ISDIR(stat.st_mode):
        yield selected, stat
        return
    with os.scandir(selected) as listing:
        entries = sorted(listing, key=lambda entry: entry.name)
    for entry in entries:
        if entry.is_dir(follow_symlinks=False):
            yield from walked(Path(entry.path))
        else:
            yield Path(entry.path), entry.stat(follow_symlinks=False)


def excluded(path: Path) -> bool:
    """Leave incomplete files and mutable status sidecars out of immutable publication."""
    patterns = ["*.tmp", "latest.jsonl", "partial-*.jsonl", ".card.lock*"]
    return (
        any(fnmatch.fnmatch(path.name, pattern) for pattern in patterns)
        or (path.parent.name == "events" and path.name == "status.json")
        or (path.parent.name == "objects" and fnmatch.fnmatch(path.name, "tmp????????"))
    )
