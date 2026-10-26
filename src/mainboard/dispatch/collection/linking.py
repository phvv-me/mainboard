"""The duplicate linker: identical settled evidence files hard-linked to one copy."""

import os
import re
import time
from collections import defaultdict
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from pathlib import Path, PurePosixPath
from stat import S_ISREG

from .pack import Digests, excluded, sha256, walked

# How long a run must have written nothing before `link` counts its files settled, in seconds.
_QUIET = 3600

_DIGEST = re.compile(r"[0-9a-f]{64}")

# A file as a walk met it: its path and its one `lstat`.
type _Entry = tuple[Path, os.stat_result]


def link(
    root: str, *, relative: str, cache: str = "", quiet: float = _QUIET, dry: bool = False
) -> dict[str, int]:
    """Hard-link identical settled files under one workspace-relative path to one copy.

    Answers how many files were seen, hashed, linked and found damaged, and the bytes their
    distinct inodes held before and after. Nothing is deleted: a duplicate's name moves onto the
    copy only after both hashed equal, a file named by a digest must hash to it, and a file
    changed since it was hashed is left alone. Files of one filesystem are grouped by size first,
    so one with no same-sized peer is never read. Settled files only: never an event stream or a
    file still written (`excluded`), nor one of a run (the directory under `artifacts`) that
    wrote anything in the last `quiet` seconds.
    cache: as `pack`'s; a linked name takes the copy's modification time, recorded there with its
        digest so collection does not hash it again.
    dry: count what would be linked, linking nothing.
    """
    base = Path(root).expanduser().resolve(strict=True)
    digests = Digests(base / cache if cache else None, base=base)
    entries = [
        (path, stat)
        for path, stat in walked(base.joinpath(*PurePosixPath(relative).parts))
        if S_ISREG(stat.st_mode)
    ]
    linking = _Linking(entries, digests=digests, dry=dry)
    copies, damaged, hashed = linking.grouped(quiet=quiet)
    done = [linking.linked(held, digest) for (_, digest), held in copies.items()]
    digests.save()
    before = sum({(stat.st_dev, stat.st_ino): stat.st_size for _, stat in entries}.values())
    return {
        "files": len(entries),
        "hashed": hashed,
        "linked": sum(linked for linked, _ in done),
        "damaged": damaged,
        "before": before,
        "after": before - sum(freed for _, freed in done),
    }


class _Linking:
    """One `link` pass over the regular files under a path, as a walk met them."""

    @staticmethod
    def _copies(
        inodes: Sequence[list[_Entry]], hashed: Sequence[str]
    ) -> tuple[dict[tuple[int, str], list[list[_Entry]]], int]:
        """The inodes by filesystem and digest, and how many were damaged.

        Damaged: a file named by a digest that its bytes do not hash to, which is never linked.
        """
        copies: dict[tuple[int, str], list[list[_Entry]]] = defaultdict(list)
        damaged = 0
        for names, digest in zip(inodes, hashed, strict=True):
            if any(_DIGEST.fullmatch(path.name) and path.name != digest for path, _ in names):
                damaged += 1
            else:
                copies[names[0][1].st_dev, digest].append(names)
        return copies, damaged

    @staticmethod
    def _relink(source: Path, *, path: Path) -> None:
        """Point `path` at `source`'s bytes through a link made beside it and moved over it."""
        staged = path.with_name(f".{path.name}.link.tmp")
        with suppress(FileNotFoundError):
            staged.unlink()
        os.link(source, staged)
        os.replace(staged, path)

    def __init__(self, entries: Sequence[_Entry], *, digests: Digests, dry: bool) -> None:
        self.entries = entries
        self.digests = digests
        self.dry = dry

    def _settled(self, *, quiet: float) -> list[list[_Entry]]:
        """The names of each inode among settled files that share a filesystem and a size.

        Settled: of a run that wrote nothing in the last `quiet` seconds, never an event stream, an
        empty file or one still written (`_is_excluded`).
        """
        newest: dict[Path, float] = defaultdict(float)
        for path, stat in self.entries:
            newest[_run(path)] = max(newest[_run(path)], stat.st_mtime)
        cutoff = time.time() - quiet
        peers: dict[tuple[int, int], dict[int, list[_Entry]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for path, stat in self.entries:
            settled = newest[_run(path)] < cutoff and path.parent.name != "events"
            if stat.st_size and settled and not excluded(path):
                peers[stat.st_dev, stat.st_size][stat.st_ino].append((path, stat))
        return [names for group in peers.values() if len(group) > 1 for names in group.values()]

    def grouped(
        self, *, quiet: float
    ) -> tuple[dict[tuple[int, str], list[list[_Entry]]], int, int]:
        """The settled inodes by filesystem and digest, how many were damaged, how many hashed.

        Only inodes sharing a filesystem and a size with another are read at all.
        """
        inodes = self._settled(quiet=quiet)
        with ThreadPoolExecutor(max_workers=16) as pool:
            hashed = list(pool.map(lambda names: sha256(names[0][0]), inodes))
        copies, damaged = self._copies(inodes, hashed)
        return copies, damaged, len(inodes)

    def linked(self, held: Sequence[list[_Entry]], digest: str) -> tuple[int, int]:
        """How many names of one digest's inodes moved onto its copy, and the bytes that freed.

        The copy kept is the one most linked already, so a rerun converges on it.
        """
        copy, *duplicates = sorted(held, key=lambda names: (-names[0][1].st_nlink, names[0][0]))
        source, kept = copy[0]
        linked = freed = 0
        for names in duplicates:
            moved = [
                path
                for path, stat in names
                if self.dry or (_is_unchanged(source, kept) and _is_unchanged(path, stat))
            ]
            for path in () if self.dry else moved:
                _Linking._relink(source, path=path)
            linked += len(moved)
            # Bytes come back once no name links them, here or anywhere outside this scan.
            freed += kept.st_size if len(moved) == len(names) == names[0][1].st_nlink else 0
        for path, _ in [*copy, *(name for names in duplicates for name in names)]:
            if not self.dry and path.lstat().st_ino == kept.st_ino:
                self.digests.record(path, kept, digest)
        return linked, freed


def _run(path: Path) -> Path:
    """The run directory `path` was written under (the one under `artifacts`), else itself."""
    parts = path.parts
    with suppress(ValueError):
        under = parts.index("artifacts")
        if under + 2 < len(parts):
            return Path(*parts[: under + 2])
    return path


def _is_unchanged(path: Path, seen: os.stat_result) -> bool:
    """Whether `path` is still the file `seen` described when it was hashed."""
    now = path.lstat()
    return (now.st_ino, now.st_size, now.st_mtime_ns) == (
        seen.st_ino,
        seen.st_size,
        seen.st_mtime_ns,
    )
