"""Walking a tree: the ignore and deny rules that prune it, its entries, and the digests
both ends of a transfer compare them by."""

import hashlib
import json
import os
import pathlib
import re
import stat
from contextlib import suppress
from typing import TYPE_CHECKING, NamedTuple, TypedDict

from .constants import DIRECTORY, FILE, LINK
from .exceptions import Refusal

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping, Sequence

    Json = str | int | bool | None | Sequence["Json"] | Mapping[str, "Json"]


# A named group inside one compiled ignore pattern, which a joined expression cannot repeat.
_NAMED = re.compile(r"(?<!\\)\(\?P<[^>]+>")


class RulesSpec(TypedDict):
    """One `Rules` as it crosses the wire."""

    patterns: dict[str, list[tuple[str, bool]]]
    paths: list[str]
    repositories: list[str]


class ScopeSpec(TypedDict):
    """One `Scope` as it crosses the wire."""

    roots: list[str]
    ignore: RulesSpec
    deny: RulesSpec
    keep: list[str]


def is_unsafe(relative: str) -> bool:
    """Whether `relative`, forward slashes only, fails to name a path strictly below a root."""
    return any(part in ("", ".", "..") for part in relative.split("/"))


def checked(relative: str) -> str:
    """`relative` unchanged when it is a safe path below a root, else a refusal naming it."""
    if is_unsafe(relative):
        raise Refusal(f"unsafe path: {relative!r}")
    return relative


def native(root: str, *, relative: str) -> str:
    """`relative`, a forward-slash workspace path, as this OS spells it under `root`."""
    return os.path.join(root, *relative.split("/"))


def sha256_of(path: str) -> str:
    """The SHA-256 of `path`'s bytes, read a chunk at a time."""
    with open(path, "rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def ancestors(path: str) -> list[str]:
    """Every directory above `path`, outermost first, the root left out."""
    parts = path.split("/")
    return ["/".join(parts[:depth]) for depth in range(1, len(parts))]


def _stamp(status: os.stat_result) -> list[int | str]:
    """What a file's digest is remembered by.

    Beside size and times, the inode tells a replaced file from an edited one.
    """
    return [status.st_size, status.st_ino, status.st_mtime_ns, status.st_ctime_ns]


class Digests:
    """File digests remembered by each file's stamp, so an unchanged file is read once.

    The stamp is the size, the inode and both nanosecond times. A write moves the change time,
    which no tool can set back, so a file rewritten to the same size under a restored mtime still
    reads as changed, and one whose stamp stands still holds the bytes it was hashed with. That
    lets both ends of a mirror compare contents without rereading a tree that did not change.

    path: the JSON file the memory lives in, created on the first save.
    """

    @staticmethod
    def _stored(path: str) -> dict[str, list[int | str]]:
        """The memory saved at `path`, empty when there is none or it does not parse."""
        try:
            return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
        except OSError, ValueError:
            return {}

    def __init__(self, path: str) -> None:
        self.path = path
        self.held = Digests._stored(path)
        self.seen: set[str] = set()

    def note(self, path: str, *, key: str, found: str) -> None:
        """Remember `found` as the digest of the file just written at `path`."""
        self.seen.add(key)
        self.held[key] = [*_stamp(os.stat(path)), found]

    def of(self, path: str, *, key: str) -> str:
        """`path`'s SHA-256, off the memory while its stamp stands."""
        stamp = _stamp(os.stat(path))
        self.seen.add(key)
        remembered = self.held.get(key)
        if remembered is not None and remembered[:-1] == stamp:
            return str(remembered[-1])
        found = sha256_of(path)
        self.held[key] = [*stamp, found]
        return found

    def save(self, *, prune: bool = False) -> None:
        """Publish the memory by rename, so a reader never meets half a document.

        prune: forget every path this memory was not asked about, which a full survey does so a
            deleted file's digest does not outlive it.
        """
        if prune:
            self.held = {key: value for key, value in self.held.items() if key in self.seen}
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        staged = f"{self.path}.{os.getpid()}.tmp"
        with open(staged, "w", encoding="utf-8", newline="\n") as out:
            json.dump(self.held, out)
        os.replace(staged, self.path)


class Rules:
    """One ordered rule set: ignore patterns anchored per directory, and literal paths.

    A pattern is a regular expression compiled on the center from one ignore line and anchored
    at the directory that declared it. The last pattern to match decides, and a path is judged
    from the root down, so a deeper ignore file overrides its parents and a negation re-includes.
    A repository boundary starts the judgement over, since a repository answers to its own
    ignore files and never to a parent's. A literal path matches itself and everything beneath
    it, which is how a results path holding glob characters still names exactly one tree.

    Each directory's patterns are joined into one expression, last pattern first, so one match
    finds the deciding pattern and a hundred-line ignore file costs one match per path. A pattern
    that names one exact path is kept out of the expression and looked up by path instead: an
    ignore file listing thousands of files one by one (research/reproducibility, 7,662) made
    the joined expression cost a pin over a minute of matching.

    patterns: base directory, "" for the root, to its ordered `(regex, verdict)` pairs.
    repositories: the bases where a repository of its own begins.
    discover: answers a base not yet known, its pairs and whether a repository begins there,
        which is how the center reads each ignore file the first time a walk reaches its
        directory; a target is handed every base the center read, up front.
    """

    @classmethod
    def of(cls, spec: RulesSpec) -> Rules:
        """The rule set a spec describes."""
        return cls(spec["patterns"], spec["paths"], repositories=spec["repositories"])

    @staticmethod
    def _literal(regex: str) -> str | None:
        """The path a pattern names exactly, None when it can match more than one spelling."""
        if not (regex.startswith("^") and regex.endswith(_LITERAL_TAIL)):
            return None
        body = regex[1 : -len(_LITERAL_TAIL)]
        if not _ESCAPED.fullmatch(body):
            return None
        return re.sub(r"\\(.)", r"\1", body, flags=re.DOTALL) or None

    @staticmethod
    def _literally(literals: Mapping[str, int], relative: str) -> int:
        """The last rule naming `relative` or a directory above it literally, -1 for none.

        A literal names itself and every path beneath it; the last rule still decides.
        """
        steps = relative.rstrip("/").split("/")
        found = (literals.get("/".join(steps[:length]), -1) for length in range(1, len(steps) + 1))
        return max(found, default=-1)

    def __deciding(self, base: str, *, candidate: str) -> int:
        """Which rule `base` declares decides `candidate`, its index, -1 when none matches.

        Asking reads `base`'s rules first, which may find a repository beginning there.
        """
        joined, _, literals = self.__declared(base)
        relative = candidate[len(base) + 1 :] if base else candidate
        found = joined.match(relative) if joined else None
        deciding = int(found.lastgroup[1:]) if found and found.lastgroup else -1
        return max(deciding, Rules._literally(literals, relative)) if literals else deciding

    def __declared(self, base: str) -> tuple[re.Pattern[str] | None, list[bool], dict[str, int]]:
        if base not in self.joined:
            rows, repository = self.discover(base) if self.discover is not None else ([], False)
            self.__learn(base, rows)
            if repository:
                self.repositories.add(base)
        return self.joined[base]

    def __init__(
        self,
        patterns: dict[str, list[tuple[str, bool]]] | None = None,
        paths: Sequence[str] = (),
        *,
        repositories: Sequence[str] = (),
        discover: Callable[[str], tuple[list[tuple[str, bool]], bool]] | None = None,
    ) -> None:
        self.patterns: dict[str, list[tuple[str, bool]]] = {}
        self.joined: dict[str, tuple[re.Pattern[str] | None, list[bool], dict[str, int]]] = {}
        for base, rows in (patterns or {}).items():
            self.__learn(base, rows)
        self.paths = tuple(paths)
        self.repositories = set(repositories)
        self.discover = discover

    def __learn(self, base: str, rows: Sequence[tuple[str, bool]]) -> None:
        """Keep `base`'s pairs as sent, and join them last first into one expression."""
        self.patterns[base] = [(regex, verdict) for regex, verdict in rows]
        literals: dict[str, int] = {}
        alternatives = []
        for index, (regex, _) in reversed(list(enumerate(rows))):
            literal = Rules._literal(regex)
            if literal is not None:
                literals.setdefault(literal, index)
            else:
                alternatives.append(f"(?P<r{index}>{_NAMED.sub('(?:', regex)})")
        self.joined[base] = (
            re.compile("|".join(alternatives)) if alternatives else None,
            [verdict for _, verdict in rows],
            literals,
        )

    def matches(self, path: str, *, directory: bool) -> bool:
        """Whether this rule set claims `path`, a directory judged with its trailing slash."""
        if any(path == literal or path.startswith(literal + "/") for literal in self.paths):
            return True
        candidate = path + "/" if directory else path
        parts = path.split("/")
        verdict = False
        for depth in range(len(parts)):
            base = "/".join(parts[:depth])
            deciding = self.__deciding(base, candidate=candidate)
            if base in self.repositories:
                verdict = False
            if deciding >= 0:
                verdict = self.joined[base][1][deciding]
        return verdict

    def read(self, base: str) -> None:
        """Learn `base`'s rules now, so a spec sent before any walk reaches it carries them."""
        self.__declared(base)

    def spec(self) -> RulesSpec:
        """This rule set as it crosses the wire, every base read so far included."""
        return {
            "patterns": {base: list(rows) for base, rows in self.patterns.items() if rows},
            "paths": list(self.paths),
            "repositories": sorted(self.repositories),
        }


# How the center spells an ignore line naming one exact path, file or directory, in `Rules`.
_LITERAL_TAIL = "(?:(?P<ps_d>/)|$)"


# A pattern body that is one exact path: escaped non-word characters and plain ones only.
_ESCAPED = re.compile(r"(?:\\(?:\W|_)|[^\\.^$*+?{}\[\]|()])*", re.DOTALL)


class Scope:
    """The trees one walk covers and the rules that prune them.

    A path is excluded when the deny rules claim it, or when the ignore rules do and it is not
    kept. Keeping is how a file a repository tracks ships whatever its ignore files say: the
    center names each such file, and a directory holding one is walked however it is ignored.
    A scope whose files are listed reads the ignore rules of every directory holding one up
    front, since a target pruning the same tree walks by nothing else.

    roots: workspace-relative starting points, each a file or a directory.
    deny: the rules that exclude whatever else says, the denylist and the host's excludes.
    keep: files the ignore rules never exclude.
    follow: walk through links as if they were what they point at, which only the center does,
        for a tree of links whose targets a host has never seen.
    listed: the files the center already knows are in scope, read from version control, which
        a walk then states instead of discovering, less what the deny rules claim; never sent,
        since a target walks its own.
    """

    @classmethod
    def of(cls, spec: ScopeSpec) -> Scope:
        """The scope a spec describes, walked without following links."""
        return cls(
            spec["roots"],
            ignore=Rules.of(spec["ignore"]),
            deny=Rules.of(spec["deny"]),
            keep=spec["keep"],
        )

    def __allowed(self, listed: Sequence[str]) -> list[str]:
        """The listed files the deny rules leave, directly and through every directory.

        Each directory is judged once, and one under a denied directory is denied without being
        judged, which is what keeps a listing of a hundred thousand files fast.
        """
        denied: dict[str, bool] = {"": False}
        return [
            path
            for path in listed
            if not self.__beneath_denied(path, denied)
            and not self.deny.matches(path, directory=False)
        ]

    def __beneath_denied(self, path: str, denied: dict[str, bool]) -> bool:
        """Whether a directory above `path` is denied, each judged once into `denied`."""
        parent = ""
        for folder in ancestors(path):
            if folder not in denied:
                denied[folder] = denied[parent] or self.deny.matches(folder, directory=True)
            if denied[folder]:
                return True
            parent = folder
        return False

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
        for folder in {"", *(folder for path in self.listed or () for folder in ancestors(path))}:
            self.ignore.read(folder)
        self.keep = set(keep)
        self.holding = {folder for path in self.keep for folder in ancestors(path)}

    def excluded(self, path: str, *, directory: bool) -> bool:
        """Whether the deny rules claim `path`, or the ignore rules do and it is not kept."""
        if self.deny.matches(path, directory=directory):
            return True
        kept = path in self.holding if directory else path in self.keep
        return not kept and self.ignore.matches(path, directory=directory)

    def spec(self) -> ScopeSpec:
        """This scope as it crosses the wire."""
        return {
            "roots": list(self.roots),
            "ignore": self.ignore.spec(),
            "deny": self.deny.spec(),
            "keep": sorted(self.keep),
        }


class Entry(NamedTuple):
    """One path a walk found, and what a transfer compares it by; its survey record in order.

    path: workspace-relative, forward slashes.
    kind: `FILE`, `DIRECTORY` or `LINK`.
    size / mtime: a file's bytes and nanosecond modification time.
    executable: a file's owner execute bit.
    detail: a file's SHA-256 once hashed, a link's target, empty for a directory.
    """

    path: str
    kind: str
    size: int = 0
    mtime: int = 0
    executable: bool = False
    detail: str = ""

    @classmethod
    def stated(cls, path: str, status: os.stat_result) -> Entry:
        """The file entry a stat describes."""
        executable = bool(status.st_mode & stat.S_IXUSR)
        return cls(
            path, kind=FILE, size=status.st_size, mtime=status.st_mtime_ns, executable=executable
        )


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
        yield from _visit(root, relative=top, scope=scope, seen=seen)


def _status(path: str, *, follow: bool) -> os.stat_result | None:
    """`path`'s status, a link's target's when following links, None when it is gone.

    A link to nothing reads as gone when following links.
    """
    try:
        status = os.lstat(path)
    except OSError:
        return None
    if not (follow and stat.S_ISLNK(status.st_mode)):
        return status
    with suppress(OSError):
        return os.stat(path)
    return None


def _stated(root: str, scope: Scope, listed: Sequence[str]) -> Iterator[Entry]:
    made: set[str] = set()
    for path in listed:
        status = _status(native(root, relative=path), follow=False)
        if status is None:
            continue
        yield from _holders(path, roots=scope.roots, made=made)
        if stat.S_ISLNK(status.st_mode):
            yield Entry(path, kind=LINK, detail=os.readlink(native(root, relative=path)))
        elif stat.S_ISREG(status.st_mode):
            yield Entry.stated(path, status)


def _holders(path: str, *, roots: Sequence[str], made: set[str]) -> Iterator[Entry]:
    """The directories above `path` inside one of `roots` not yet yielded, each noted in `made`."""
    for folder in ancestors(path):
        if folder not in made and any(
            folder == top or folder.startswith(top + "/") for top in roots
        ):
            made.add(folder)
            yield Entry(folder, kind=DIRECTORY)


def _visit(
    root: str, *, relative: str, scope: Scope, seen: set[tuple[int, int]]
) -> Iterator[Entry]:
    path = native(root, relative=relative)
    status = _status(path, follow=scope.follow)
    directory = status is not None and stat.S_ISDIR(status.st_mode)
    if status is None or scope.excluded(relative, directory=directory):
        return
    if stat.S_ISLNK(status.st_mode):
        yield Entry(relative, kind=LINK, detail=os.readlink(path))
    elif stat.S_ISREG(status.st_mode):
        yield Entry.stated(relative, status)
    elif directory and not (scope.follow and (status.st_dev, status.st_ino) in seen):
        seen.add((status.st_dev, status.st_ino))
        yield Entry(relative, kind=DIRECTORY)
        for name in sorted(os.listdir(path)):
            yield from _visit(root, relative=f"{relative}/{name}", scope=scope, seen=seen)


def digested(entry: Entry, root: str, digests: Digests) -> Entry | None:
    """`entry` with its digest when it is a file, None when the file vanished before its read."""
    if entry.kind == FILE:
        try:
            return entry._replace(
                detail=digests.of(native(root, relative=entry.path), key=entry.path)
            )
        except FileNotFoundError:
            return None
    return entry
