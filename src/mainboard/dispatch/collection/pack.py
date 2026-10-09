"""Standard-library evidence upkeep on a host, run by mainboard's own Python there: the result
exporter (`pack`), sent over SSH stdin, and the duplicate linker (`link`)."""

import fnmatch
import hashlib
import json
import os
import re
import sys
import time
from collections import defaultdict
from collections.abc import Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from pathlib import Path, PurePosixPath
from stat import S_ISDIR, S_ISREG
from types import MappingProxyType
from zipfile import ZIP_ZSTANDARD, ZipFile

# The archive's last member when any file was left unsent because the center's lake keeps its
# bytes, as {path: digest}. Outside every collected scope, so no evidence file can take its name.
HELD = "mainboard-held.json"

# The zstd level every member is compressed at. Measured on 1.5 GB of cutok evidence on a GH200
# node (2026-10-09): level 1 sent 3.5x fewer bytes for 3.7 s of one core, level 3 3.9x for 7.5 s,
# level 6 4.3x for 16.6 s; at a few MB/s over ssh, level 3's bytes save more than its CPU costs.
_LEVEL = 3

# How long a run must have written nothing before `link` counts its files settled, in seconds.
_QUIET = 3600

_DIGEST = re.compile(r"[0-9a-f]{64}")


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
    named: dict[str, str] = {}
    with ZipFile(sys.stdout.buffer, "w", ZIP_ZSTANDARD, compresslevel=_LEVEL) as archive:
        for path, stat in _paths(base, relative):
            if path.parent.name == "events" and path.name == "live.ndjson":
                _snapshot(archive, path, base=base, known=known)
                continue
            key = path.relative_to(base).as_posix()
            digest = digests.of(path, stat)
            if known.get(key) == digest:
                continue
            if digest in held:
                named[key] = digest
            else:
                _immutable(archive, path, base=base)
        if named:
            archive.writestr(HELD, json.dumps(named))
    digests.save()


def link(
    root: str, *, relative: str, cache: str = "", quiet: float = _QUIET, dry: bool = False
) -> dict[str, int]:
    """Hard-link identical settled files under one workspace-relative path to one copy.

    Answers how many files were seen, hashed, linked and found damaged, and the bytes their
    distinct inodes held before and after. Nothing is deleted: a duplicate's name moves onto the
    copy only after both hashed equal, a file named by a digest must hash to it, and a file
    changed since it was hashed is left alone. Files of one filesystem are grouped by size first,
    so one with no same-sized peer is never read. Settled files only: never an event stream or a
    file still written (`_is_excluded`), nor one of a run (the directory under `artifacts`) that
    wrote anything in the last `quiet` seconds.
    cache: as `pack`'s; a linked name takes the copy's modification time, recorded there with its
        digest so collection does not hash it again.
    dry: count what would be linked, linking nothing.
    """
    base = Path(root).expanduser().resolve(strict=True)
    digests = Digests(base / cache if cache else None, base=base)
    entries = [
        (path, stat)
        for path, stat in _entries(base.joinpath(*PurePosixPath(relative).parts))
        if S_ISREG(stat.st_mode)
    ]
    newest: dict[Path, float] = defaultdict(float)
    for path, stat in entries:
        newest[_run(path)] = max(newest[_run(path)], stat.st_mtime)
    cutoff = time.time() - quiet
    peers: dict[tuple[int, int], dict[int, list[tuple[Path, os.stat_result]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for path, stat in entries:
        settled = newest[_run(path)] < cutoff and path.parent.name != "events"
        if stat.st_size and settled and not _is_excluded(path):
            peers[stat.st_dev, stat.st_size][stat.st_ino].append((path, stat))
    inodes = [names for group in peers.values() if len(group) > 1 for names in group.values()]
    with ThreadPoolExecutor(max_workers=16) as pool:
        hashed = list(pool.map(lambda names: _sha256(names[0][0]), inodes))
    copies: dict[tuple[int, str], list[list[tuple[Path, os.stat_result]]]] = defaultdict(list)
    damaged = 0
    for names, digest in zip(inodes, hashed, strict=True):
        if any(_DIGEST.fullmatch(path.name) and path.name != digest for path, _ in names):
            damaged += 1
        else:
            copies[names[0][1].st_dev, digest].append(names)
    linked = freed = 0
    for (_, digest), held in copies.items():
        # The copy kept is the one most linked already, so a rerun converges on it.
        copy, *duplicates = sorted(held, key=lambda names: (-names[0][1].st_nlink, names[0][0]))
        source, kept = copy[0]
        for names in duplicates:
            moved = [
                path
                for path, stat in names
                if dry or (_is_unchanged(source, kept) and _is_unchanged(path, stat))
            ]
            for path in () if dry else moved:
                _relink(source, path)
            linked += len(moved)
            # Bytes come back once no name links them, here or anywhere outside this scan.
            freed += kept.st_size if len(moved) == len(names) == names[0][1].st_nlink else 0
        for path, _ in [*copy, *(name for names in duplicates for name in names)]:
            if not dry and path.lstat().st_ino == kept.st_ino:
                digests.record(path, kept, digest)
    digests.save()
    before = sum({(stat.st_dev, stat.st_ino): stat.st_size for _, stat in entries}.values())
    return {
        "files": len(entries),
        "hashed": len(inodes),
        "linked": linked,
        "damaged": damaged,
        "before": before,
        "after": before - freed,
    }


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
        return self.record(path, stat, _sha256(path))

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


def _sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _run(path: Path) -> Path:
    """The run directory `path` was written under (the one under `artifacts`), else itself."""
    parts = path.parts
    with suppress(ValueError):
        at = parts.index("artifacts")
        if at + 2 < len(parts):
            return Path(*parts[: at + 2])
    return path


def _relink(source: Path, path: Path) -> None:
    """Point `path` at `source`'s bytes through a link made beside it and moved over it."""
    staged = path.with_name(f".{path.name}.link.tmp")
    with suppress(FileNotFoundError):
        staged.unlink()
    os.link(source, staged)
    os.replace(staged, path)


def _is_unchanged(path: Path, seen: os.stat_result) -> bool:
    """Whether `path` is still the file `seen` described when it was hashed."""
    now = path.lstat()
    return (now.st_ino, now.st_size, now.st_mtime_ns) == (
        seen.st_ino,
        seen.st_size,
        seen.st_mtime_ns,
    )


def _paths(base: Path, relative: str) -> Iterator[tuple[Path, os.stat_result]]:
    """Enumerate contained regular files with their one `lstat`, retaining traversal failures.

    The selection is checked to lie inside the workspace once; below it nothing is followed,
    so every regular file found lies inside too, and any link refuses the collection.
    """
    selected = base.joinpath(*PurePosixPath(relative).parts)
    if not selected.resolve(strict=True).is_relative_to(base):
        raise ValueError("collection path escapes the remote workspace")
    for path, stat in _entries(selected):
        if _is_excluded(path):
            continue
        if not S_ISREG(stat.st_mode):
            raise ValueError("collection contains a non-regular file: " + str(path))
        yield path, stat


def _entries(selected: Path) -> Iterator[tuple[Path, os.stat_result]]:
    """Walk a directory by `scandir`, one `lstat` per entry, a link yielded as itself."""
    stat = selected.lstat()
    if not S_ISDIR(stat.st_mode):
        yield selected, stat
        return
    with os.scandir(selected) as listing:
        entries = sorted(listing, key=lambda entry: entry.name)
    for entry in entries:
        if entry.is_dir(follow_symlinks=False):
            yield from _entries(Path(entry.path))
        else:
            yield Path(entry.path), entry.stat(follow_symlinks=False)


def _is_excluded(path: Path) -> bool:
    """Leave incomplete files and mutable status sidecars out of immutable publication."""
    patterns = ["*.tmp", "latest.jsonl", "partial-*.jsonl", ".card.lock*"]
    return (
        any(fnmatch.fnmatch(path.name, pattern) for pattern in patterns)
        or (path.parent.name == "events" and path.name == "status.json")
        or (path.parent.name == "objects" and fnmatch.fnmatch(path.name, "tmp????????"))
    )


def _immutable(archive: ZipFile, path: Path, *, base: Path) -> None:
    """Copy one file and refuse an observed size or timestamp change."""
    before = path.stat()
    archive.write(path, path.relative_to(base).as_posix())
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError("file changed during collection: " + str(path))


def _snapshot(archive: ZipFile, path: Path, *, base: Path, known: Mapping[str, str]) -> None:
    """Retain complete event records byte-for-byte under their validated offset range.

    A range the center already collected is not sent again.
    """
    with path.open("rb") as stream:
        data = stream.read(path.stat().st_size)
    data = data[: data.rfind(b"\n") + 1]
    if not data:
        return
    start, end = _range(data)
    name = path.with_name(f"collected-{start:020d}-{end:020d}.ndjson").relative_to(base)
    if name.as_posix() not in known:
        archive.writestr(name.as_posix(), data)


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
