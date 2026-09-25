# What a mirror carries and what it may never touch: every repository's own view of its files,
# the permanent denylist, the host's own filter patterns, and the lock that serializes one
# target's mirrors.
#
# The file set is decided per repository the way git decides it: where git answers, each
# repository lists every file it tracks, whatever an ignore file says, and the untracked files
# its own ignore files leave. A parent's ignore file never reaches into a nested repository,
# which is what once dropped a submodule's tracked `build/` sources under the monorepo's
# `build/` rule and broke the host's Rust build. Without git the same ignore files are read
# directly, repository boundaries included. The host's excludes and the denylist apply on top.

import hashlib
import shutil
import subprocess  # ruff:ignore[suspicious-subprocess-import]  reason=git argv built from typed fields, not untrusted input since=2026-09-25
from fnmatch import fnmatchcase
from functools import cached_property
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Self

import pathspec
from filelock import FileLock
from patos import FrozenModel

from ..core.errors import MissionError
from .agent import Rules, Scope, walk
from .agent.program import FILE, LINK
from .shared import state_dir, state_path
from .transport import Endpoint

if TYPE_CHECKING:
    from collections.abc import Sequence
    from types import TracebackType

# Always skipped regardless of `.gitignore`, in the filter pattern language `patterns` reads.
# `.git` has no trailing slash so it also matches a submodule's `.git` file, not just the
# superproject's `.git/` directory.
ALWAYS_EXCLUDE = (
    ".git",
    ".env",
    f"{state_dir()}/",
    ".mainboard/",
    ".pixi/",
    "__pycache__/",
    # Published trial output travels down only. Uploading a fetched live segment replaces
    # the remote writer's open inode and loses its subsequent events.
    "*/evidence/artifacts/***",
    "*/evidence/receipts/***",
)

# Host/device coordination is never transferable, even under an explicit include rule, and a
# mirror never removes one a host holds.
CARD_LEASES = (".card.lock", ".card.lock.*")


def compiled(lines: Sequence[str]) -> list[tuple[str, bool]]:
    """Ignore-file lines as the `(regex, verdict)` pairs a `Rules` base holds, in order."""
    # pyrefly: ignore  reason=pathspec from_lines stub over-narrows to AnyStr since=2026-08-16
    spec = pathspec.GitIgnoreSpec.from_lines(lines)
    return [
        (pattern.regex.pattern, bool(pattern.include))
        for pattern in spec.patterns
        if pattern.include is not None and pattern.regex is not None
    ]


def patterns(declared: Sequence[str], *, paths: Sequence[str] = ()) -> Rules:
    """Filter patterns, as `[hosts.*.sync]` and the denylist write them, as one root rule set.

    The language is gitignore's with the two readings a mirror's patterns have always had. A
    pattern is anchored to the workspace root only when it starts with `/`, so `data/raw` names
    every `data/raw` in the tree, and `dir/***` names a directory and everything beneath it.

    declared: the patterns, every one of which excludes what it matches.
    paths: literal workspace-relative paths, each matching itself and everything beneath it.
    """
    lines = []
    for pattern in declared:
        written = pattern.removesuffix("/***") + "/" if pattern.endswith("/***") else pattern
        floating = "/" in written.rstrip("/") and not written.startswith(("/", "**/"))
        lines.append(f"**/{written}" if floating else written)
    return Rules({"": compiled(lines)}, paths)


def denied(excluded: Sequence[str] = (), *, paths: Sequence[str] = ()) -> Rules:
    """What a mirror never ships: the denylist, the host's `excluded` patterns, the card leases.

    paths: literal workspace-relative paths denied with everything beneath them.
    """
    return patterns([*ALWAYS_EXCLUDE, *excluded, *CARD_LEASES], paths=paths)


class Listing(FrozenModel):
    """What version control says a tree holds.

    files: every file in scope, tracked or untracked and not ignored.
    kept: the tracked files the repositories' own ignore files would drop, shipped all the same.
    """

    files: tuple[str, ...]
    kept: tuple[str, ...]


class SyncLock:
    """Serialize mirror-and-pin transactions to one target across local dispatch processes.

    target: the effective rental endpoint or declared SSH alias being mirrored.
    root: local workspace root that owns the generated state directory, discovered upward from
        the current directory when None, so two processes started in different subdirectories
        still queue behind the same lock file.
    """

    def __init__(self, target: str | Endpoint, root: Path | None = None) -> None:
        identity = (
            target if isinstance(target, str) else f"{target.destination}:{target.port or 22}"
        )
        digest = hashlib.blake2s(identity.encode(), digest_size=8).hexdigest()
        self.path = state_path(root) / "locks" / f"sync-{digest}.lock"
        # A mirror nested inside a dispatch's transaction shares its reentrant lock instance.
        self.lock = FileLock(self.path.resolve(), is_singleton=True)

    def __enter__(self) -> Self:
        """Wait for this target's mirror lock and hold it until context exit."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock.acquire()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Release the kernel lock even when the mirror raises."""
        if self.lock.is_locked:
            self.lock.release()


class GitignoreFilter:
    """The workspace's ignore files as one rule set, and its repositories' own file lists.

    Read the way git reads them: every `.gitignore` applies from its own directory, a deeper one
    overriding its parents, and a repository's `info/exclude` and the user's global excludes join
    at its root, where a parent's rules stop. Each file is read once, when a walk or a question
    first reaches its directory, and the compiled rules travel to a target as they are, so a host
    prunes by exactly the rules shipped and needs no ignore parser of its own.

    root: the repo whose ignore files decide, the current working directory by default.
    """

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or Path.cwd()
        self.rules = Rules(discover=self.__declared)
        self.__indexes: dict[str, tuple[tuple[int, int, int], list[str], list[str]]] = {}

    @staticmethod
    def validate_sources(paths: Sequence[str]) -> None:
        """Refuse explicit sources that cannot cross a host's coordination boundary."""
        leases = [
            path
            for path in paths
            if any(
                fnmatchcase(part, pattern)
                for part in PurePosixPath(path).parts
                for pattern in CARD_LEASES
            )
        ]
        if leases:
            raise ValueError(f"card leases cannot be declared as transferable source: {leases}")

    def ignored(self, path: str | Path) -> bool:
        """Whether the ignore files claim `path`, without invoking version control."""
        candidate = Path(path)
        if candidate.is_absolute() and candidate.is_relative_to(self.root):
            candidate = candidate.relative_to(self.root)
        directory = (self.root / candidate).is_dir()
        return self.rules.matches(candidate.as_posix(), directory=directory)

    def scope(self, roots: Sequence[str], *, deny: Rules) -> Scope:
        """The workspace tree under `roots`, as a mirror ships it and a host prunes it.

        roots: the workspace-relative paths the tree starts from.
        deny: what is excluded whatever the repositories say.
        """
        listing = self.tracked(roots, deny=deny)
        if listing is None:
            return Scope(roots, ignore=self.rules, deny=deny)
        return Scope(roots, ignore=self.rules, deny=deny, keep=listing.kept, listed=listing.files)

    def files(self, roots: Sequence[str], *, excluded: Sequence[str] = ()) -> list[str]:
        """The source files under `roots` a mirror ships, a link counted when it leads to a file.

        excluded: the host's own sync excludes, joining the denylist.
        """
        scope = self.scope(roots, deny=denied(excluded))
        return sorted(
            entry.path
            for entry in walk(str(self.root), scope)
            if entry.kind == FILE or (entry.kind == LINK and (self.root / entry.path).is_file())
        )

    def tracked(self, roots: Sequence[str], *, deny: Rules | None = None) -> Listing | None:
        """Every file its repository would call source under `roots`, or None without git.

        Each repository answers for itself, a registered submodule or a nested clone alike. A
        path it tracks but no longer holds is left out by the walk that states it. The tracked
        files its ignore files would drop are listed as `kept`, which a target is told to keep.

        deny: what is excluded whatever the repositories say, so a repository nested where it
            claims is never asked at all.
        """
        if self.__git is None or not (self.root / ".git").exists():
            return None
        found: list[str] = []
        kept: list[str] = []
        pending = [""]
        while pending:
            repository = pending.pop()
            specs = _within(repository, roots)
            if specs is not None and not (deny and deny.matches(repository, directory=True)):
                files, ignored, nested = self.__repository(repository, specs)
                found += files
                kept += ignored
                pending += nested
        return Listing(files=_under(found, roots), kept=_under(kept, roots))

    def __repository(
        self, repository: str, specs: list[str]
    ) -> tuple[list[str], list[str], list[str]]:
        """One repository's files, those it tracks but ignores, and the repositories nested in
        it, as workspace paths.

        Tracked files come from its index whole; untracked ones from a walk narrowed to `specs`,
        which is not cheap, and a pathspec reaching into a submodule is left to that submodule,
        since git refuses one.

        repository: workspace-relative root, "" for the workspace itself.
        specs: the pathspecs inside it that are in scope, empty for all of it.
        """
        indexed, links = self.__indexed(repository)
        files = list(indexed)
        nested = [link for link in links if (self.root / repository / link / ".git").exists()]
        narrowed = [
            spec
            for spec in specs
            if not any(spec == link or spec.startswith(link + "/") for link in links)
        ]
        ignored: list[str] = []
        if narrowed or not specs:
            for path in self.__listed(repository, ["--others", "--exclude-standard"], narrowed):
                (nested if path.endswith("/") else files).append(path.rstrip("/"))
            standard = ["--cached", "--ignored", "--exclude-standard"]
            ignored = self.__listed(repository, standard, narrowed)
        return (
            [_joined(repository, path) for path in files],
            [_joined(repository, path) for path in ignored],
            [_joined(repository, path) for path in nested],
        )

    def __indexed(self, repository: str) -> tuple[list[str], list[str]]:
        """What `repository`'s index tracks, files apart from submodule links.

        Remembered against the index file's stamp, so a dispatch reading several roots reads a
        large index once, and a `git add` in between is seen.
        """
        index = (_gitdir(self.root / repository) or self.root / repository / ".git") / "index"
        try:
            status = index.stat()
        except FileNotFoundError:
            return [], []
        stamp = (status.st_size, status.st_mtime_ns, status.st_ino)
        remembered = self.__indexes.get(repository)
        if remembered is not None and remembered[0] == stamp:
            return remembered[1], remembered[2]
        files: list[str] = []
        links: list[str] = []
        for row in self.__listed(repository, ["--stage"]):
            meta, _, path = row.partition("\t")
            (links if meta.startswith("160000 ") else files).append(path)
        self.__indexes[repository] = (stamp, files, links)
        return files, links

    @cached_property
    def __git(self) -> str | None:
        """The git this machine answers with, None when it has none."""
        return shutil.which("git")

    @cached_property
    def __excludes(self) -> list[tuple[str, bool]]:
        """The user's global excludes, which join every repository's rules at its root."""
        configured = self.__run(
            ["config", "--path", "--get", "core.excludesFile"], check=False
        ).strip()
        return _read(
            Path(configured) if configured else Path.home() / ".config" / "git" / "ignore"
        )

    def __listed(
        self, repository: str, options: Sequence[str], specs: Sequence[str] = ()
    ) -> list[str]:
        """`git ls-files` in `repository` under `specs`, one entry per NUL-separated record."""
        where = str(self.root / repository)
        output = self.__run(["-C", where, "ls-files", "-z", *options, "--", *specs])
        return [record for record in output.split("\0") if record]

    def __run(self, arguments: list[str], *, check: bool = True) -> str:
        """One git command's output; a failing one refuses by what git said, unless unchecked."""
        answered = subprocess.run(  # ruff:ignore[subprocess-without-shell-equals-true]  reason=git argv built from typed fields since=2026-09-25
            [self.__git or "git", "--literal-pathspecs", *arguments],
            capture_output=True,
            check=False,
        )
        if check and answered.returncode:
            said = answered.stderr.decode("utf-8", errors="replace").strip()
            raise MissionError(f"git could not list the workspace's files: {said}")
        return answered.stdout.decode("utf-8", errors="surrogateescape")

    def __declared(self, base: str) -> tuple[list[tuple[str, bool]], bool]:
        """The compiled rules that apply from `base` down, and whether a repository starts there.

        A repository root carries the user's global excludes and its own `info/exclude` ahead
        of its `.gitignore`, which is the order git lets them override each other in.
        """
        directory = self.root / base
        rows = _read(directory / ".gitignore")
        gitdir = _gitdir(directory)
        if gitdir is None:
            return rows, False
        global_rows = self.__excludes if self.__git is not None else []
        return [*global_rows, *_read(gitdir / "info" / "exclude"), *rows], True


def _read(ignore: Path) -> list[tuple[str, bool]]:
    """The compiled rules of one ignore file, none when there is no such file."""
    try:
        return compiled(ignore.read_text(encoding="utf-8").splitlines())
    except FileNotFoundError, NotADirectoryError, IsADirectoryError:
        return []


def _gitdir(directory: Path) -> Path | None:
    """The git directory of the repository rooted at `directory`, None when none starts there.

    A submodule's `.git` is a file naming its git directory elsewhere, relative to it.
    """
    marker = directory / ".git"
    if marker.is_dir():
        return marker
    if not marker.is_file():
        return None
    named = marker.read_text(encoding="utf-8").strip().removeprefix("gitdir:").strip()
    return directory / named


def _under(paths: Sequence[str], roots: Sequence[str]) -> tuple[str, ...]:
    """`paths` that lie at or below one of `roots`, sorted and without repeats."""
    exact, beneath = set(roots), tuple(f"{top}/" for top in roots)
    return tuple(sorted(path for path in set(paths) if path in exact or path.startswith(beneath)))


def _joined(repository: str, path: str) -> str:
    """`path` inside `repository`, as a workspace-relative path."""
    return f"{repository}/{path}" if repository else path


def _within(repository: str, roots: Sequence[str]) -> list[str] | None:
    """The pathspecs that narrow `repository`'s untracked listing to `roots`.

    Empty when a root holds the whole repository, None when no root reaches into it at all.
    """
    if not repository:
        return list(roots)
    if any(repository == top or repository.startswith(top + "/") for top in roots):
        return []
    inside = [top[len(repository) + 1 :] for top in roots if top.startswith(repository + "/")]
    return inside or None
