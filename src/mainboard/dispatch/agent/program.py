"""The standard-library agent both ends of a transfer run.

The center imports this module to walk and hash its own workspace, and sends this very source to
a target over SSH, where any Python from 3.9 on runs it with nothing installed: the target
reports what it holds, takes what changed as one tar stream, prunes what the center's rules say
to prune, and pins snapshots under a kernel file lock. Nothing here imports beyond the standard
library or uses syntax newer than 3.9, since the interpreter on the far side is whatever the
machine shipped with.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import re
import shutil
import stat
import sys
import tarfile
import tempfile
from contextlib import contextmanager
from typing import IO, TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence

    Json = str | int | bool | None | Sequence["Json"] | Mapping[str, "Json"]

# How much of a file is read or copied at a time, so no file ever has to fit in memory.
CHUNK = 1 << 20

# A pinned tree's own control files, which no closure may ship and no live path may replace.
STAMP = ".mainboard-source"
CLOSURE = ".mainboard-closure"
WRAPPERS = ".mainboard-jobs"

# Whether this interpreter stands on Windows, read once so a test can stand in for either OS.
WINDOWS = os.name == "nt"

# What one walked path is: a regular file, a directory, or a link carried as a link.
FILE, DIRECTORY, LINK = "f", "d", "l"

# A named group inside one compiled ignore pattern, which a joined expression cannot repeat.
_NAMED = re.compile(r"(?<!\\)\(\?P<[^>]+>")

# A memory that cannot be read is an empty one; a tuple so the handler stays one clause.
_UNREADABLE = (OSError, ValueError)


class RulesSpec(TypedDict):
    """One `Rules` as it crosses the wire: compiled patterns per base, literal paths, and the
    bases where a repository of its own begins."""

    patterns: dict[str, list[tuple[str, bool]]]
    paths: list[str]
    repositories: list[str]


class ScopeSpec(TypedDict):
    """One `Scope` as it crosses the wire."""

    roots: list[str]
    ignore: RulesSpec
    deny: RulesSpec
    keep: list[str]


class SurveySpec(TypedDict):
    """What a target is asked to describe: its scopes walked and its named files stated."""

    root: str
    state: str
    scopes: list[ScopeSpec]
    named: list[str]


class ReceiveSpec(TypedDict):
    """What a target is told to change before and while the tar stream arrives.

    files: each file the stream carries, as `[mtime_ns, executable]`, executable None when the
        center has no mode bits to give.
    """

    root: str
    state: str
    delete: list[str]
    directories: list[str]
    links: dict[str, str]
    files: dict[str, tuple[int, bool | None]]


class ImageSpec(TypedDict, total=False):
    """What a snapshot copies: a mirrored scope, or a sealed closure listing and its needs."""

    kind: str
    scope: ScopeSpec
    listing: str
    digest: str
    needs: list[str]
    pins: list[str]
    staging: str
    live: list[str]


class PinSpec(TypedDict):
    """Everything one pin needs, resolved on the center down to host path strings."""

    root: str
    base: str
    key: str
    stamp: str
    out: str
    prefix: str
    environment: str
    results: str
    script: str
    wrapper: str
    image: ImageSpec


class Request(TypedDict, total=False):
    """One request, naming its operation by the one key it carries."""

    survey: SurveySpec
    receive: ReceiveSpec
    pin: PinSpec


class Refusal(Exception):
    """A request this end will not carry out, reported as one line rather than a traceback."""


def unsafe(relative: str) -> bool:
    """Whether `relative` fails to name a path strictly below a root on this OS.

    Forward slashes only. On Windows a backslash or a colon inside a name would climb out of the
    root or name a drive or a stream, so neither is part of a name there.
    """
    parts = relative.split("/")
    return any(part in ("", ".", "..") for part in parts) or (
        WINDOWS and any(mark in relative for mark in "\\:")
    )


def checked(relative: str) -> str:
    """`relative` unchanged when it is a safe path below a root, else a refusal naming it."""
    if unsafe(relative):
        raise Refusal(f"unsafe path: {relative!r}")
    return relative


def native(root: str, relative: str) -> str:
    """`relative`, a forward-slash workspace path, as this OS spells it under `root`."""
    return os.path.join(root, *relative.split("/"))


def digest(path: str) -> str:
    """The SHA-256 of `path`'s bytes, read a chunk at a time."""
    hashed = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(CHUNK), b""):
            hashed.update(chunk)
    return hashed.hexdigest()


def containers(sources: Sequence[str]) -> list[str]:
    """Every directory a mirrored snapshot creates to hold `sources`, the tree root included.

    A snapshot fills each of these with links back to the mirror for whatever it did not copy,
    which is how a job reaches the environment, the dispatch state and the data directories
    beside its own source without any of them being copied.
    """
    found = {"."}
    for source in sources:
        parts = source.split("/")[:-1]
        found.update("/".join(parts[: depth + 1]) for depth in range(len(parts)))
    return sorted(found)


class Digests:
    """File digests remembered by each file's stamp, so an unchanged file is read once.

    The stamp is the size, the inode and both nanosecond times. A write moves the change time,
    which no tool can set back, so a file rewritten to the same size under a restored
    modification time still reads as changed, and one whose stamp stands still holds the bytes it
    was hashed with. That is what lets both ends of a mirror compare contents without rereading
    a tree that did not change.

    path: the JSON file the memory lives in, created on the first save.
    """

    def __init__(self, path: str) -> None:
        self.path = path
        try:
            with open(path, encoding="utf-8") as stored:
                self.held: dict[str, list[int | str]] = json.load(stored)
        except _UNREADABLE:
            self.held = {}
        self.seen: set[str] = set()

    def of(self, path: str, *, key: str) -> str:
        """`path`'s SHA-256, read off the memory while its stamp stands."""
        stamp = _stamp(os.stat(path))
        self.seen.add(key)
        remembered = self.held.get(key)
        if remembered is not None and remembered[:-1] == stamp:
            return str(remembered[-1])
        found = digest(path)
        self.held[key] = [*stamp, found]
        return found

    def note(self, path: str, *, key: str, found: str) -> None:
        """Remember `found` as the digest of the file just written at `path`."""
        self.seen.add(key)
        self.held[key] = [*_stamp(os.stat(path)), found]

    def save(self, *, prune: bool = False) -> None:
        """Publish the memory by rename, so a reader never meets half a document.

        prune: forget every path this memory was not asked about, which a full survey does so a
            deleted file's digest does not outlive it.
        """
        if prune:
            self.held = {key: value for key, value in self.held.items() if key in self.seen}
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        staged = f"{self.path}.{os.getpid()}.tmp"
        with open(staged, "w", encoding="utf-8") as out:
            json.dump(self.held, out)
        os.replace(staged, self.path)


def _stamp(status: os.stat_result) -> list[int | str]:
    """What a file's digest is remembered by: its size, inode and both nanosecond times."""
    return [status.st_size, status.st_ino, status.st_mtime_ns, status.st_ctime_ns]


class Rules:
    """One ordered rule set: ignore patterns anchored per directory, and literal paths.

    A pattern is a regular expression compiled on the center from one ignore line and anchored
    at the directory that declared it. The last pattern to match decides, and a path is judged
    from the root down, so a deeper ignore file overrides its parents and a negation re-includes.
    A repository boundary starts the judgement over, since a repository answers to its own
    ignore files and never to a parent's. A literal path matches itself and everything beneath
    it, which is how a results path holding glob characters still names exactly one tree.

    Each directory's patterns are joined into one expression, last pattern first, so a single
    match finds the pattern that decides; a root ignore file of a hundred lines then costs one
    match per path rather than a hundred.

    patterns: base directory, "" for the root, to its ordered `(regex, verdict)` pairs.
    paths: literal workspace-relative paths.
    repositories: the bases where a repository of its own begins.
    discover: answers a base not yet known, its pairs and whether a repository begins there,
        which is how the center reads each ignore file the first time a walk reaches its
        directory; a target is handed every base the center read, up front.
    """

    def __init__(
        self,
        patterns: dict[str, list[tuple[str, bool]]] | None = None,
        paths: Sequence[str] = (),
        repositories: Sequence[str] = (),
        discover: Callable[[str], tuple[list[tuple[str, bool]], bool]] | None = None,
    ) -> None:
        self.patterns: dict[str, list[tuple[str, bool]]] = {}
        self.joined: dict[str, tuple[re.Pattern[str] | None, list[bool]]] = {}
        for base, rows in (patterns or {}).items():
            self.__learn(base, rows)
        self.paths = tuple(paths)
        self.repositories = set(repositories)
        self.discover = discover

    @classmethod
    def of(cls, spec: RulesSpec) -> Rules:
        """The rule set a spec describes."""
        return cls(spec["patterns"], spec["paths"], spec["repositories"])

    def spec(self) -> RulesSpec:
        """This rule set as it crosses the wire, every base read so far included."""
        return {
            "patterns": {
                base: [(regex, verdict) for regex, verdict in rows]
                for base, rows in self.patterns.items()
                if rows
            },
            "paths": list(self.paths),
            "repositories": sorted(self.repositories),
        }

    def matches(self, path: str, *, directory: bool) -> bool:
        """Whether this rule set claims `path`, a directory judged with its trailing slash."""
        if any(path == literal or path.startswith(literal + "/") for literal in self.paths):
            return True
        candidate = path + "/" if directory else path
        parts = path.split("/")
        verdict = False
        for depth in range(len(parts)):
            base = "/".join(parts[:depth])
            joined, verdicts = self.__declared(base)
            if base in self.repositories:
                verdict = False
            found = (
                joined.match(candidate[len(base) + 1 :] if base else candidate) if joined else None
            )
            if found is not None and found.lastgroup is not None:
                verdict = verdicts[int(found.lastgroup[1:])]
        return verdict

    def read(self, base: str) -> None:
        """Learn `base`'s rules now, so a spec sent before any walk reaches it carries them."""
        self.__declared(base)

    def __declared(self, base: str) -> tuple[re.Pattern[str] | None, list[bool]]:
        if base not in self.joined:
            rows, repository = self.discover(base) if self.discover is not None else ([], False)
            self.__learn(base, rows)
            if repository:
                self.repositories.add(base)
        return self.joined[base]

    def __learn(self, base: str, rows: Sequence[tuple[str, bool]]) -> None:
        """Keep `base`'s pairs as sent, and join them last first into one expression."""
        self.patterns[base] = [(regex, verdict) for regex, verdict in rows]
        alternatives = [
            f"(?P<r{index}>{_NAMED.sub('(?:', regex)})"
            for index, (regex, _) in reversed(list(enumerate(rows)))
        ]
        joined = re.compile("|".join(alternatives)) if alternatives else None
        self.joined[base] = (joined, [verdict for _, verdict in rows])


class Scope:
    """The trees one walk covers and the rules that prune them.

    A path is excluded when the deny rules claim it, or when the ignore rules do and it is not
    kept. Keeping is how a file a repository tracks ships whatever its ignore files say: the
    center names each such file, and a directory holding one is walked however it is ignored.
    A scope whose files are listed reads the ignore rules of every directory holding one up
    front, since a target pruning the same tree walks by nothing else.

    roots: workspace-relative starting points, each a file or a directory.
    ignore: the ignore files' rules.
    deny: the rules that exclude whatever else says, the denylist and the host's excludes.
    keep: files the ignore rules never exclude.
    follow: walk through links as if they were what they point at, which only the center does,
        for a tree of links whose targets a host has never seen.
    listed: the files the center already knows are in scope, read from version control, which
        a walk then states instead of discovering, less what the deny rules claim; never sent,
        since a target walks its own.
    """

    def __init__(
        self,
        roots: Sequence[str],
        *,
        ignore: Rules | None = None,
        deny: Rules | None = None,
        keep: Sequence[str] = (),
        follow: bool = False,
        listed: Sequence[str] | None = None,
    ) -> None:
        self.roots = tuple(roots)
        self.ignore = ignore or Rules()
        self.deny = deny or Rules()
        self.follow = follow
        self.listed = None if listed is None else self.__allowed(listed)
        for folder in _folders(self.listed or ()):
            self.ignore.read(folder)
        self.keep = set(keep)
        self.holding = {
            "/".join(parts[:depth])
            for parts in (path.split("/") for path in self.keep)
            for depth in range(1, len(parts))
        }

    @classmethod
    def of(cls, spec: ScopeSpec) -> Scope:
        """The scope a spec describes, walked without following links."""
        return cls(
            spec["roots"],
            ignore=Rules.of(spec["ignore"]),
            deny=Rules.of(spec["deny"]),
            keep=spec["keep"],
        )

    def spec(self) -> ScopeSpec:
        """This scope as it crosses the wire."""
        return {
            "roots": list(self.roots),
            "ignore": self.ignore.spec(),
            "deny": self.deny.spec(),
            "keep": sorted(self.keep),
        }

    def excluded(self, path: str, *, directory: bool) -> bool:
        """Whether the deny rules claim `path`, or the ignore rules do and it is not kept."""
        if self.deny.matches(path, directory=directory):
            return True
        kept = path in self.holding if directory else path in self.keep
        return not kept and self.ignore.matches(path, directory=directory)

    def __allowed(self, listed: Sequence[str]) -> list[str]:
        """The listed files the deny rules leave, directly and through every directory.

        Each directory is judged once, and one under a denied directory is denied without being
        judged, which is what keeps a listing of a hundred thousand files fast.
        """
        denied: dict[str, bool] = {"": False}
        allowed = []
        for path in listed:
            parts = path.split("/")
            claimed = False
            for depth in range(1, len(parts)):
                folder = "/".join(parts[:depth])
                if folder not in denied:
                    parent = "/".join(parts[: depth - 1])
                    denied[folder] = denied[parent] or self.deny.matches(folder, directory=True)
                claimed = denied[folder]
                if claimed:
                    break
            if not claimed and not self.deny.matches(path, directory=False):
                allowed.append(path)
        return allowed


def _folders(paths: Iterable[str]) -> set[str]:
    """Every directory holding one of `paths`, the root included."""
    found = {""}
    for path in paths:
        parts = path.split("/")
        found.update("/".join(parts[:depth]) for depth in range(1, len(parts)))
    return found


class Entry:
    """One path a walk found, and what a transfer compares it by.

    path: workspace-relative, forward slashes.
    kind: `FILE`, `DIRECTORY` or `LINK`.
    size / mtime: a file's bytes and nanosecond modification time.
    executable: a file's owner execute bit, None where the file system keeps none.
    detail: a file's SHA-256 once hashed, a link's target, empty for a directory.
    """

    __slots__ = ("detail", "executable", "kind", "mtime", "path", "size")

    def __init__(
        self,
        path: str,
        kind: str,
        size: int = 0,
        mtime: int = 0,
        executable: bool | None = None,
        detail: str = "",
    ) -> None:
        self.path = path
        self.kind = kind
        self.size = size
        self.mtime = mtime
        self.executable = executable
        self.detail = detail

    @classmethod
    def stated(cls, path: str, status: os.stat_result) -> Entry:
        """The file entry a stat of `path` describes."""
        executable = None if WINDOWS else bool(status.st_mode & stat.S_IXUSR)
        return cls(path, FILE, status.st_size, status.st_mtime_ns, executable)

    def record(self) -> list[Json]:
        """This entry as one survey record."""
        return [self.path, self.kind, self.size, self.mtime, self.executable, self.detail]


def walk(root: str, scope: Scope) -> Iterator[Entry]:
    """Every entry under `scope`'s roots below `root`, an excluded directory never entered.

    A root that is not there yields nothing. A link is yielded as a link unless the scope
    follows links, when it is walked as what it points at, a link to nothing is skipped and a
    directory already walked is not walked twice. A scope whose files are listed is stated
    rather than walked: each listed file, and the directories that hold them below a root.
    """
    if scope.listed is not None:
        yield from _stated(root, scope, scope.listed)
        return
    seen: set[tuple[int, int]] = set()
    for top in scope.roots:
        yield from _visit(root, top, scope, seen)


def _stated(root: str, scope: Scope, listed: Sequence[str]) -> Iterator[Entry]:
    made: set[str] = set()
    for path in listed:
        parts = path.split("/")
        folders = ["/".join(parts[:depth]) for depth in range(1, len(parts))]
        try:
            status = os.lstat(native(root, path))
        except OSError:
            continue
        for folder in folders:
            if folder not in made and any(
                folder == top or folder.startswith(top + "/") for top in scope.roots
            ):
                made.add(folder)
                yield Entry(folder, DIRECTORY)
        if stat.S_ISLNK(status.st_mode):
            yield Entry(path, LINK, detail=os.readlink(native(root, path)))
        elif stat.S_ISREG(status.st_mode):
            yield Entry.stated(path, status)


def _visit(root: str, relative: str, scope: Scope, seen: set[tuple[int, int]]) -> Iterator[Entry]:
    path = native(root, relative)
    try:
        status = os.lstat(path)
        if stat.S_ISLNK(status.st_mode) and scope.follow:
            status = os.stat(path)
    except OSError:
        return
    if stat.S_ISLNK(status.st_mode):
        if not scope.excluded(relative, directory=False):
            yield Entry(relative, LINK, detail=os.readlink(path))
        return
    directory = stat.S_ISDIR(status.st_mode)
    if scope.excluded(relative, directory=directory):
        return
    if not directory:
        if stat.S_ISREG(status.st_mode):
            yield Entry.stated(relative, status)
        return
    if scope.follow:
        if (status.st_dev, status.st_ino) in seen:
            return
        seen.add((status.st_dev, status.st_ino))
    yield Entry(relative, DIRECTORY)
    for name in sorted(os.listdir(path)):
        yield from _visit(root, f"{relative}/{name}", scope, seen)


def hashed(entry: Entry, root: str, digests: Digests) -> Entry | None:
    """`entry` with its digest when it is a file, None when the file vanished before its read."""
    if entry.kind == FILE:
        try:
            entry.detail = digests.of(native(root, entry.path), key=entry.path)
        except FileNotFoundError:
            return None
    return entry


@contextmanager
def locked(path: str) -> Iterator[None]:
    """Hold an exclusive kernel lock on `path` for the block, released however it ends.

    A kernel lock rather than a lock file's mere existence, so a process killed mid-transfer
    leaves nothing behind that the next one would have to be told about.
    """
    with open(path, "a+b") as handle:
        if WINDOWS:
            msvcrt = importlib.import_module("msvcrt")
            handle.seek(0)
            while True:
                try:
                    msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
                    break
                except OSError:
                    continue
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            fcntl = importlib.import_module("fcntl")
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class Emitter:
    """Answer records to the center, one JSON line each, flushed so a quiet link means a stall."""

    def __init__(self, stdout: IO[bytes]) -> None:
        self.stdout = stdout

    def __call__(self, record: Json) -> None:
        self.stdout.write(json.dumps(record, separators=(",", ":")).encode("utf-8") + b"\n")
        self.stdout.flush()


def survey(spec: SurveySpec, emit: Emitter) -> None:
    """Describe what this end holds: its capabilities first, then every scoped and named entry.

    The root and its state directory are made on the way, since a fresh machine holds neither
    and this is the first thing a mirror asks of it.
    """
    root = spec["root"]
    state = native(root, spec["state"])
    os.makedirs(state, exist_ok=True)
    emit({"links": not WINDOWS, "modes": not WINDOWS, "fold": _folds(state)})
    digests = Digests(os.path.join(state, "digests.json"))
    for scope in spec["scopes"]:
        for entry in walk(root, Scope.of(scope)):
            found = hashed(entry, root, digests)
            if found is not None:
                emit(found.record())
    for relative in spec["named"]:
        try:
            status = os.lstat(native(root, checked(relative)))
        except FileNotFoundError:
            continue
        if not stat.S_ISREG(status.st_mode):
            continue
        found = hashed(Entry.stated(relative, status), root, digests)
        if found is not None:
            emit(found.record())
    digests.save(prune=True)


def _folds(directory: str) -> bool:
    """Whether names in `directory` compare without case, probed with a file of its own."""
    handle, probe = tempfile.mkstemp(prefix=".mainboard-Case-", dir=directory)
    os.close(handle)
    try:
        head, name = os.path.split(probe)
        return os.path.exists(os.path.join(head, name.swapcase()))
    finally:
        os.remove(probe)


def receive(spec: ReceiveSpec, stream: IO[bytes], emit: Emitter) -> None:
    """Apply one mirror under this end's mirror lock: prune, make, link, then unpack the stream.

    Pruning comes first so a name that only changed case lands on a file system that folds case,
    and every file is written beside its final name and renamed over it, so a reader holding the
    old file keeps the bytes it opened.
    """
    root = spec["root"]
    state = native(root, spec["state"])
    os.makedirs(state, exist_ok=True)
    with locked(os.path.join(state, "mirror.lock")):
        deleted, kept = _prune(root, spec["delete"])
        for relative in spec["directories"]:
            os.makedirs(native(root, checked(relative)), exist_ok=True)
        for relative, target in spec["links"].items():
            path = native(root, checked(relative))
            os.makedirs(os.path.dirname(path), exist_ok=True)
            os.symlink(target, path)
        digests = Digests(os.path.join(state, "digests.json"))
        written, size = _unpack(root, stream, spec["files"], digests)
        digests.save()
    emit({"written": written, "bytes": size, "deleted": deleted, "kept": kept})


def _prune(root: str, paths: Sequence[str]) -> tuple[list[str], list[str]]:
    """Remove `paths` deepest first; a directory still holding what the rules keep stays."""
    deleted: list[str] = []
    kept: list[str] = []
    for relative in sorted(paths, key=lambda path: path.count("/"), reverse=True):
        path = native(root, checked(relative))
        try:
            if os.path.isdir(path) and not os.path.islink(path):
                os.rmdir(path)
            else:
                os.remove(path)
        except FileNotFoundError:
            continue
        except OSError:
            kept.append(relative)
            continue
        deleted.append(relative)
    return deleted, kept


def _unpack(
    root: str, stream: IO[bytes], files: dict[str, tuple[int, bool | None]], digests: Digests
) -> tuple[int, int]:
    """Write every file the stream carries, refusing one the request did not announce."""
    written = size = 0
    with tarfile.open(fileobj=stream, mode="r|gz") as archive:
        for member in archive:
            relative = checked(member.name)
            source = archive.extractfile(member)
            if relative not in files or source is None:
                raise Refusal(f"unannounced entry in the stream: {relative!r}")
            mtime, executable = files[relative]
            path = native(root, relative)
            digests.note(path, key=relative, found=_place(path, source, mtime, executable))
            written += 1
            size += member.size
    return written, size


def _place(path: str, source: IO[bytes], mtime: int, executable: bool | None) -> str:
    """Write `source` beside `path`, rename it into place, and answer the digest of its bytes.

    The file is created with the mode its execute bit asks for, which this host's umask then
    narrows, and in binary mode, since Windows would otherwise turn every newline it writes into
    a carriage return and a newline.
    """
    parent = os.path.dirname(path)
    staged = os.path.join(parent, f".{os.path.basename(path)}.mainboard-{os.getpid()}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_BINARY", 0)
    hashed = hashlib.sha256()
    try:
        os.makedirs(parent, exist_ok=True)
        with os.fdopen(os.open(staged, flags, 0o777 if executable else 0o666), "wb") as out:
            for chunk in iter(lambda: source.read(CHUNK), b""):
                hashed.update(chunk)
                out.write(chunk)
        os.utime(staged, ns=(mtime, mtime))
        os.replace(staged, path)
    except OSError as fault:
        if os.path.lexists(staged):
            os.remove(staged)
        raise Refusal(f"could not place {path}: {fault}") from fault
    return hashed.hexdigest()


def _remove(path: str) -> None:
    """Remove whatever stands at `path`, a link itself rather than what it points at."""
    if os.path.islink(path) or os.path.isfile(path):
        os.remove(path)
    elif os.path.isdir(path):
        shutil.rmtree(path)


def _relink(source: str, path: str) -> None:
    """Point a link at `path` to `source`, replacing a link already there."""
    if os.path.islink(path):
        os.remove(path)
    os.symlink(source, path)


def _hardlink(source: str, path: str) -> None:
    """Share `source`'s inode at `path`, or copy it where the file system cannot share one."""
    try:
        os.link(source, path)
    except OSError:
        shutil.copy2(source, path)


class Mirrored:
    """The whole synced scope, what a command that ships the mirror runs from.

    scope: the roots the mirror shipped and the rules it shipped them by, so the snapshot holds
        the shipped file set and not the artifacts the host wrote beside it.
    """

    def __init__(self, spec: ImageSpec) -> None:
        self.scope = Scope.of(spec["scope"])

    def copy(self, root: str, snap: str) -> None:
        """Hardlink the shipped set out of the mirror, links carried as links."""
        for top in self.scope.roots:
            if not os.path.lexists(native(root, top)):
                raise Refusal(f"missing source: {top}")
        for entry in walk(root, self.scope):
            placed = native(snap, entry.path)
            os.makedirs(os.path.dirname(placed), exist_ok=True)
            if entry.kind == DIRECTORY:
                os.makedirs(placed, exist_ok=True)
            elif entry.kind == LINK:
                os.symlink(entry.detail, placed)
            else:
                _hardlink(native(root, entry.path), placed)

    def fill(self, root: str, snap: str) -> None:
        """Link back whatever the mirror holds in each directory the copy made and did not fill."""
        for directory in containers(self.scope.roots):
            source, target = native(root, directory), native(snap, directory)
            os.makedirs(target, exist_ok=True)
            for name in sorted(os.listdir(source)):
                entry, placed = os.path.join(source, name), os.path.join(target, name)
                if os.path.exists(entry) and not os.path.lexists(placed):
                    os.symlink(entry, placed)

    def verify(self, snap: str) -> None:
        """Nothing: a mirrored tree carries no listing to check."""

    def link(self, root: str, snap: str) -> None:
        """Nothing: a mirrored tree already reaches everything the mirror holds."""


class Sealed:
    """A job's closure and nothing beside it, what a job spelled by file runs from.

    listing: the closure listing on the mirror, whose first column names every shipped file and
        whose second holds each file's SHA-256.
    digest: the listing's own SHA-256, which the frozen copy must match.
    needs / pins: data paths linked back to the mirror on every dispatch, pins reached through
        their shared `staging` directory.
    live: every path linked back on this dispatch, which no listed file may sit beneath.
    """

    def __init__(self, spec: ImageSpec) -> None:
        self.listing = spec["listing"]
        self.digest = spec["digest"]
        self.needs = spec["needs"]
        self.pins = spec["pins"]
        self.staging = spec["staging"]
        self.live = spec["live"]

    def copy(self, root: str, snap: str) -> None:
        """Freeze the listing, then hardlink exactly the files it names, links resolved."""
        frozen = os.path.join(snap, CLOSURE)
        shutil.copyfile(native(root, checked(self.listing)), frozen)
        for relative, _ in self.__rows(frozen):
            source = native(root, relative)
            if not os.path.isfile(source):
                raise Refusal(f"missing or linked source: {relative}")
            placed = native(snap, relative)
            os.makedirs(os.path.dirname(placed), exist_ok=True)
            _hardlink(os.path.realpath(source), placed)

    def fill(self, root: str, snap: str) -> None:
        """Nothing: a sealed tree reaches the mirror only through what it declared."""

    def verify(self, snap: str) -> None:
        """Check the frozen listing and every listed file's place and bytes."""
        real = os.path.realpath(snap)
        for relative, blob in self.__rows(os.path.join(snap, CLOSURE)):
            for live in self.live:
                if relative == live or relative.startswith(live + "/"):
                    raise Refusal(f"live path overlaps source: {live}")
            path = native(snap, relative)
            if os.path.islink(path) or not os.path.isfile(path):
                raise Refusal(f"missing or linked source: {relative}")
            if os.path.realpath(path) != native(real, relative):
                raise Refusal(f"linked source parent: {relative}")
            if digest(path) != blob:
                raise Refusal(f"source blob mismatch: {relative}")

    def link(self, root: str, snap: str) -> None:
        """Check each pin and need on the mirror and link it into the tree."""
        for pin in self.pins:
            if not os.path.exists(native(root, pin)):
                raise Refusal(f"the need {pin} is not on the mirror")
        if self.pins and not os.path.exists(native(snap, self.staging)):
            os.makedirs(os.path.dirname(native(snap, self.staging)), exist_ok=True)
            _relink(native(root, self.staging), native(snap, self.staging))
        for need in self.needs:
            if not os.path.exists(native(root, need)):
                raise Refusal(f"the need {need} is not on the mirror")
            os.makedirs(os.path.dirname(native(snap, need)), exist_ok=True)
            _relink(native(root, need), native(snap, need))

    def __rows(self, frozen: str) -> Iterator[tuple[str, str]]:
        """Each listed path and blob, the listing's own digest and every path checked first."""
        if digest(frozen) != self.digest:
            raise Refusal("closure listing digest mismatch")
        with open(frozen, encoding="utf-8", newline="") as listing:
            rows = [line.split("\t") for line in listing.read().split("\n") if line]
        for fields in rows:
            relative = fields[0]
            if unsafe(relative):
                raise Refusal(f"invalid closure path: {relative}")
            if relative in (CLOSURE, STAMP, WRAPPERS) or relative.startswith(WRAPPERS + "/"):
                raise Refusal(f"reserved closure path: {relative}")
            yield relative, fields[1]


class Snapshot:
    """One pinned source tree: built once under the host's pin lock, verified on every reuse.

    A build happens in a private directory that becomes the tree by one rename after its source
    verifies, so a failed build never becomes a completed snapshot, and an older incomplete tree
    refuses for inspection. The live paths, the needs, the results path and the frozen wrapper,
    belong to the dispatch rather than to the tree, so they are linked on every reuse too.
    """

    def __init__(self, spec: PinSpec) -> None:
        self.spec = spec
        self.root = spec["root"]
        self.base = spec["base"]
        self.final = os.path.join(self.base, spec["key"])
        image = spec["image"]
        self.image: Mirrored | Sealed = (
            Sealed(image) if image["kind"] == "sealed" else Mirrored(image)
        )

    def pinned(self) -> str:
        """Build or verify the tree under the pin lock, and answer where it stands."""
        os.makedirs(self.base, exist_ok=True)
        with locked(os.path.join(self.base, ".pin.lock")):
            if os.path.isfile(os.path.join(self.final, STAMP)):
                self.__reuse()
            else:
                self.__build()
        return self.final

    def __build(self) -> None:
        if os.path.lexists(self.final):
            raise Refusal(f"incomplete snapshot requires inspection: {self.final}")
        pending = tempfile.mkdtemp(prefix=".pending.", dir=self.base)
        try:
            self.image.copy(self.root, pending)
            self.__generated(pending)
            self.image.fill(self.root, pending)
            self.__environment(pending)
            self.image.verify(pending)
            self.__live(pending)
            with open(os.path.join(pending, STAMP), "wb") as stamp:
                stamp.write(self.spec["stamp"].encode("utf-8"))
            os.rename(pending, self.final)
        finally:
            if os.path.lexists(pending):
                shutil.rmtree(pending)

    def __reuse(self) -> None:
        with open(os.path.join(self.final, STAMP), "rb") as stamp:
            if stamp.read() != self.spec["stamp"].encode("utf-8"):
                raise Refusal("snapshot stamp mismatch")
        self.image.verify(self.final)
        self.__live(self.final)

    def __live(self, snap: str) -> None:
        self.image.link(self.root, snap)
        self.__results(snap)
        self.__wrapper(snap)

    def __generated(self, snap: str) -> None:
        """Rebuild the generated tree as this snapshot's own.

        The environment is the mirror's and the code is the snapshot's, and the generated tree is
        where the two meet, so its directories down to each environment are real here, its files
        hardlinked, and everything heavy beneath them a link back to the mirror. A job that
        recompiles its manifest then writes into this tree and never over the description every
        other job on the host activates through. A directory the image already copied into is
        left as the image made it.
        """
        out = self.spec["out"]
        envs = native(self.root, f"{out}/envs")
        names = sorted(os.listdir(envs)) if os.path.isdir(envs) else []
        os.makedirs(native(snap, f"{out}/envs"), exist_ok=True)
        for name in names:
            if os.path.isdir(os.path.join(envs, name)):
                os.makedirs(native(snap, f"{out}/envs/{name}"), exist_ok=True)
        for directory in [out, *(f"{out}/envs/{name}" for name in names)]:
            source = native(self.root, directory)
            if not os.path.isdir(source):
                continue
            for name in sorted(os.listdir(source)):
                entry = os.path.join(source, name)
                placed = native(snap, f"{directory}/{name}")
                if not os.path.exists(entry) or os.path.lexists(placed):
                    continue
                if os.path.isdir(entry):
                    os.symlink(entry, placed)
                else:
                    _hardlink(entry, placed)

    def __environment(self, snap: str) -> None:
        """Point the tree's environment at the immutable prefix it names, once, at build time."""
        prefix = self.spec["prefix"]
        if not prefix:
            return
        where = native(snap, f"{self.spec['out']}/envs/{self.spec['environment']}")
        os.makedirs(where, exist_ok=True)
        _remove(os.path.join(where, ".pixi"))
        os.symlink(f"{prefix}/.pixi", os.path.join(where, ".pixi"))

    def __results(self, snap: str) -> None:
        """Point the dispatch's results path back at the mirror, where the pull already looks.

        A real directory the copy made there is replaced by the link. When the path's parent is
        itself reached through a link into the mirror, the path already is the mirror's and is
        left alone, since clearing it would clear the mirror's own results.
        """
        relative = self.spec["results"]
        if not relative:
            return
        os.makedirs(native(self.root, relative), exist_ok=True)
        placed = native(snap, relative)
        parent = os.path.dirname(placed)
        os.makedirs(parent, exist_ok=True)
        inside = native(os.path.realpath(snap), os.path.dirname(relative) or ".")
        if os.path.realpath(parent) != os.path.normpath(inside):
            return
        if not os.path.islink(placed):
            _remove(placed)
        _relink(native(self.root, relative), placed)

    def __wrapper(self, snap: str) -> None:
        """Freeze the staged job wrapper by its bytes, then check the frozen copy on every pin."""
        staged, where = self.spec["script"], self.spec["wrapper"]
        if not staged:
            return
        expected = where.rsplit("/", 1)[-1][len("job-") : -len(".sh")]
        wrappers = os.path.join(snap, WRAPPERS)
        if os.path.islink(wrappers):
            raise Refusal("linked wrapper directory")
        os.makedirs(wrappers, exist_ok=True)
        frozen = native(snap, where)
        if not os.path.lexists(frozen):
            handle, pending = tempfile.mkstemp(prefix=".pending.", dir=wrappers)
            os.close(handle)
            try:
                source = native(self.root, checked(staged))
                shutil.copyfile(source, pending)
                shutil.copymode(source, pending)
                if digest(pending) != expected:
                    raise Refusal("wrapper digest mismatch")
                os.replace(pending, frozen)
            finally:
                if os.path.lexists(pending):
                    os.remove(pending)
        if os.path.islink(frozen) or not os.path.isfile(frozen):
            raise Refusal("missing or linked wrapper")
        if digest(frozen) != expected:
            raise Refusal("wrapper digest mismatch")


def run(stdin: IO[bytes], stdout: IO[bytes], stderr: IO[str]) -> int:
    """Carry out the one request `stdin` holds, answering records on `stdout`; the exit status.

    A refusal is one `mainboard:` line on `stderr` and status 3; anything else escapes as the
    traceback it is.
    """
    request: Request = json.loads(stdin.readline())
    emit = Emitter(stdout)
    try:
        if "survey" in request:
            survey(request["survey"], emit)
        elif "receive" in request:
            receive(request["receive"], stdin, emit)
        else:
            emit({"path": Snapshot(request["pin"]).pinned()})
    except Refusal as refusal:
        stderr.write(f"mainboard: {refusal}\n")
        stderr.flush()
        return 3
    return 0


if __name__ == "__main__":  # pragma: no cover - the entry a target runs, never imported there
    sys.exit(run(sys.stdin.buffer, sys.stdout.buffer, sys.stderr))
