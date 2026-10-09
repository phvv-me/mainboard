"""Standard-library result exporter, also sent over SSH stdin to mainboard's Python on a host."""

import fnmatch
import hashlib
import json
import os
import sys
from collections.abc import Iterator, Mapping
from pathlib import Path, PurePosixPath
from stat import S_ISREG
from types import MappingProxyType
from zipfile import ZIP_STORED, ZipFile


def pack(
    root: str,
    *,
    relative: str,
    known: Mapping[str, str] = MappingProxyType({}),
    cache: str = "",
) -> None:
    """Stream regular files under one workspace-relative path without following links.

    cache: workspace-relative file of remote digests, so a known file is hashed again only
        after its size or modification time changed.
    """
    base = Path(root).expanduser().resolve(strict=True)
    digests = Digests(base / cache if cache else None, base=base)
    with ZipFile(sys.stdout.buffer, "w", compression=ZIP_STORED) as archive:
        for path in _paths(base, relative):
            expected = known.get(path.relative_to(base).as_posix())
            if expected is not None and digests.of(path) == expected:
                continue
            if path.parent.name == "events" and path.name == "live.ndjson":
                _snapshot(archive, path, base=base)
            else:
                _immutable(archive, path, base=base)
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

    def of(self, path: Path) -> str:
        stat = path.stat()
        key = path.relative_to(self.base).as_posix()
        held = self.held.get(key)
        if held and held[:2] == [stat.st_size, stat.st_mtime_ns]:
            return held[2]
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        self.held[key] = [stat.st_size, stat.st_mtime_ns, digest.hexdigest()]
        self.changed = True
        return digest.hexdigest()

    def save(self) -> None:
        """Write the cache atomically, so a concurrent pass reads the old one or the new one."""
        if not (self.path and self.changed):
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        staged = self.path.with_name(f"{self.path.name}.{os.getpid()}.tmp")
        staged.write_text(json.dumps(self.held), encoding="utf-8")
        os.replace(staged, self.path)


def _paths(base: Path, relative: str) -> Iterator[Path]:
    """Enumerate contained regular files, retaining traversal and permission failures."""
    selected = base.joinpath(*PurePosixPath(relative).parts)
    if not selected.resolve(strict=True).is_relative_to(base):
        raise ValueError("collection path escapes the remote workspace")
    for path in _entries(selected):
        if _is_excluded(path):
            continue
        if not S_ISREG(path.lstat().st_mode) or not path.resolve(strict=True).is_relative_to(base):
            raise ValueError("collection contains a non-regular file: " + str(path))
        yield path


def _entries(selected: Path) -> Iterator[Path]:
    """Walk a directory without silently dropping symlink directories."""
    if selected.is_symlink() or not selected.is_dir():
        yield selected
        return
    for directory, folders, names in os.walk(selected, followlinks=False, onerror=_raise):
        links = [name for name in folders if (Path(directory) / name).is_symlink()]
        yield from (Path(directory) / name for name in sorted([*names, *links]))


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


def _snapshot(archive: ZipFile, path: Path, *, base: Path) -> None:
    """Retain complete event records byte-for-byte under their validated offset range."""
    with path.open("rb") as stream:
        data = stream.read(path.stat().st_size)
    data = data[: data.rfind(b"\n") + 1]
    if not data:
        return
    start, end = _range(data)
    name = path.with_name(f"collected-{start:020d}-{end:020d}.ndjson")
    archive.writestr(name.relative_to(base).as_posix(), data)


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


def _raise(error: OSError) -> None:
    raise error
