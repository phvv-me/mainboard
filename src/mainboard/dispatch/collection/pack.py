"""What a host runs for collection: `pack` streams its results home and `evict` removes the copies
the center's lake holds. Sent over ssh stdin as this file's text and run by mainboard's own Python
there, so it imports the standard library alone."""

import fnmatch
import hashlib
import json
import os
import sys
from collections.abc import Iterator, Mapping
from contextlib import suppress
from pathlib import Path, PurePosixPath
from stat import S_ISDIR, S_ISREG
from types import MappingProxyType
from typing import NamedTuple
from zipfile import ZIP_ZSTANDARD, ZipFile

# The archive's last member when any file was left unsent because the center's lake keeps its
# bytes, as {path: digest}. Outside every collected scope, so no evidence file can take its name.
# Defined here, not in `constants`: this file is sent to the host as the script itself, and imports
# nothing but the standard library.
HELD = "mainboard-held.json"

# The archive's member when the pass stopped at its allowance with files still to send.
MORE = "mainboard-more.json"

# The zstd level every member is compressed at. Measured on 1.5 GB of cutok evidence on a GH200
# node (2026-10-09): level 1 sent 3.5x fewer bytes for 3.7 s of one core, level 3 3.9x for 7.5 s,
# level 6 4.3x for 16.6 s; at a few MB/s over ssh, level 3's bytes save more than its CPU costs.
_LEVEL = 3


class Center(NamedTuple):
    """What the center already holds and how much one pass may bring it.

    kept: the digest the center's lake indexes at each workspace-relative path. A file there with
        those bytes is not sent, and neither is any file holding a digest the lake keeps: it is
        named in `HELD`, and the center indexes its path from the object it already has.
    known: digests of files the center holds on disk outside its lake, by path, not sent again.
    allowance: bytes of file data this pass may send, what the center can stage. The pass stops
        before the file that would pass it, marks the archive `MORE` and the center asks again; a
        single file past the allowance is refused.
    """

    kept: Mapping[str, str] = MappingProxyType({})
    known: Mapping[str, str] = MappingProxyType({})
    allowance: int = sys.maxsize


def pack(root: str, *, relative: str, center: Center, cache: str = "") -> None:
    """Stream regular files under one workspace-relative path without following links.

    cache: workspace-relative file of remote digests, so a file is hashed again only after its
        size or modification time changed.
    """
    base = Path(root).expanduser().resolve(strict=True)
    digests = Digests(base / cache if cache else None, base=base)
    known, held = {**center.kept, **center.known}, set(center.kept.values())
    with ZipFile(sys.stdout.buffer, "w", ZIP_ZSTANDARD, compresslevel=_LEVEL) as archive:
        writer = _Writer(
            archive, digests=digests, held=held, known=known, allowance=center.allowance
        )
        for path, stat in _paths(base, relative):
            writer.add(path, stat)
            if writer.spent:
                break
        if writer.named:
            archive.writestr(HELD, json.dumps(writer.named))
        if writer.spent:
            archive.writestr(MORE, "{}")
    digests.save()


def evict(root: str, *, relative: str, center: Center, cache: str = "") -> None:
    """Remove each file under one workspace-relative path that `center.kept` names with its digest.

    Prints what went and what stayed, and why, as one JSON object. Nothing is followed: a
    selection that is the workspace, lies outside it or is reached through a link is refused, a
    link below it stays a link, and a file goes only while it is still the file just hashed. Of
    the directories, only those the removals emptied go.
    center: its `kept` lists only paths whose objects the lake read back intact.
    cache: as `pack`'s, losing the entries of removed files.
    """
    base = Path(root).expanduser().resolve(strict=True)
    selected = _Eviction.selection(base, relative)
    digests = Digests(base / cache if cache else None, base=base)
    eviction = _Eviction(digests, kept=center.kept)
    try:
        files = list(walked(selected))
    except FileNotFoundError:
        # A results path the job never wrote, or one an earlier pass emptied: nothing to remove.
        files = []
    for path, stat in files:
        eviction.consider(path, stat)
    eviction.prune(selected)
    digests.save()
    json.dump(
        {"removed": eviction.removed, "kept": eviction.stayed, "emptied": eviction.emptied},
        sys.stdout,
    )


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

    def forget(self, path: Path) -> None:
        """Drop the digest of `path`, a file now gone."""
        if self.held.pop(path.relative_to(self.base).as_posix(), None) is not None:
            self.changed = True

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


class _Eviction:
    """One `evict` pass: the files removed, those kept with why, and the directories emptied."""

    @staticmethod
    def selection(base: Path, relative: str) -> Path:
        """The path `relative` names inside the workspace `base`, which need not exist.

        Refused: the workspace itself, a path outside it, and one a link stands on the way to,
        since a removal there would act past the link.
        """
        selected = base.joinpath(*PurePosixPath(relative).parts)
        if selected == base or not selected.is_relative_to(base) or selected.resolve() != selected:
            raise ValueError(
                f"eviction refuses {relative!r}: name a path inside the workspace, never the "
                "workspace itself, outside it or through a link"
            )
        return selected

    def __init__(self, digests: Digests, *, kept: Mapping[str, str]) -> None:
        self.digests = digests
        self.kept = kept
        self.removed: dict[str, int] = {}
        self.stayed: dict[str, str] = {}
        self.emptied: list[str] = []

    def _why(self, path: Path, stat: os.stat_result, key: str) -> str:
        """Why `path` stays, empty when the lake holds these very bytes at its path."""
        if not S_ISREG(stat.st_mode):
            return "a link or special file, never followed"
        if (wanted := self.kept.get(key)) is None:
            return "the lake keeps no intact copy at this path"
        if self.digests.of(path, stat) != wanted:
            return "the lake holds other bytes at this path"
        return "" if is_unchanged(path, stat) else "changed while it was checked"

    def consider(self, path: Path, stat: os.stat_result) -> None:
        """Remove `path` when the lake holds its bytes at its path, else record why it stays."""
        key = path.relative_to(self.digests.base).as_posix()
        try:
            why = self._why(path, stat, key)
        except FileNotFoundError:
            why = "gone before it was checked"
        if why:
            self.stayed[key] = why
            return
        path.unlink()
        self.digests.forget(path)
        self.removed[key] = stat.st_size

    def prune(self, selected: Path) -> None:
        """Remove each directory at or under `selected` the removals emptied, deepest first.

        `rmdir` refuses a directory that still holds anything, so none with a file goes.
        """
        base = self.digests.base
        touched = {
            folder
            for key in self.removed
            for folder in (base / key).parents
            if folder.is_relative_to(selected)
        }
        for folder in sorted(touched, key=lambda folder: len(folder.parts), reverse=True):
            with suppress(OSError):
                folder.rmdir()
                self.emptied.append(folder.relative_to(base).as_posix())


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
        digests: Digests,
        held: set[str],
        known: Mapping[str, str],
        allowance: int,
    ) -> None:
        self.archive = archive
        self.digests = digests
        self.held = held
        self.known = known
        self.allowance = allowance
        self.named: dict[str, str] = {}
        self.sent = 0
        self.spent = False

    def _immutable(self, path: Path) -> None:
        """Copy one file and refuse an observed size or timestamp change."""
        before = path.stat()
        self.archive.write(path, path.relative_to(self.digests.base).as_posix())
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError("file changed during collection: " + str(path))

    def _send(self, path: Path, stat: os.stat_result) -> None:
        """Copy one file if the pass's allowance still holds it, else end the pass before it."""
        if stat.st_size > self.allowance:
            raise ValueError(
                f"{path} holds {stat.st_size} bytes, more than the {self.allowance} the center "
                "can stage: free disk on the center first"
            )
        if self.sent + stat.st_size > self.allowance:
            self.spent = True
            return
        self._immutable(path)
        self.sent += stat.st_size

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
        name = path.with_name(f"collected-{start:020d}-{end:020d}.ndjson")
        name = name.relative_to(self.digests.base)
        if name.as_posix() not in self.known:
            self.archive.writestr(name.as_posix(), data)

    def add(self, path: Path, stat: os.stat_result) -> None:
        """Send `path` unless the center already holds its bytes."""
        if path.parent.name == "events" and path.name == "live.ndjson":
            self._snapshot(path)
            return
        key = path.relative_to(self.digests.base).as_posix()
        digest = self.digests.of(path, stat)
        if self.known.get(key) == digest:
            return
        if digest in self.held:
            self.named[key] = digest
        else:
            self._send(path, stat)


def sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def is_unchanged(path: Path, seen: os.stat_result) -> bool:
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
