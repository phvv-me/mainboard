# What a job still has to put on its target, and what that costs on the wire.
#
# A host that was onboarded already carries the workspace, and every dispatch since refreshed it,
# so the honest answer to "what does this job ship" is never the whole tree. It is the files that
# changed since that mirror was last brought up to date, plus whatever data the job itself names,
# which the mirror's include scope may not carry at all.

from compression import zstd
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pathspec
from patos import FrozenModel

from ..dispatch import sync
from ..dispatch.sync import GitignoreFilter

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from ..board import Board
    from .spec import BatchJob

# How much file is read at a time while measuring. Large enough that a big tree is bound by the
# disk rather than by the loop, small enough that one file never has to fit in memory.
_CHUNK = 1 << 20

# How many zstd streams a measurement runs at once. Measured over 600 MB of this workspace on
# eight cores: 1.57x at two threads, 2.35x at four, 2.76x at eight, 2.66x at sixteen. Eight is
# where the curve turns, and past it the extra threads only take work away from each other.
_STREAMS = 8


class TransferSet(FrozenModel):
    """What one job still has to ship, measured rather than guessed.

    job: the job this set belongs to.
    target: the alias it ships to.
    paths: the declared roots it was measured over, the mirror's include scope and the job's own
        data, so a surprising size can be traced back to what was counted.
    files: how many files are in flight.
    raw_bytes: their size on disk.
    wire_bytes: their size compressed, which is what the mirror actually sends, since every
        transfer this tool makes is compressed. Measured over a few parallel zstd streams, the
        shape rsync ships in rather than one long-lived context over the whole set.
    since: the mirror watermark the delta was measured against, empty when the target has no
        recorded mirror and everything in scope is therefore in flight.
    """

    job: str
    target: str
    paths: tuple[str, ...] = ()
    files: int = 0
    raw_bytes: int = 0
    wire_bytes: int = 0
    since: str = ""


class Transfer:
    """Measures what each job must still put on its target, one job at a time.

    Every path is read through the same board the dispatch itself uses, so the scope measured
    here is the scope that would actually ship: the host profile's include list, the workspace's
    own ignore rules, and the generated directories the mirror never carries.
    """

    def __init__(self, board: Board, *, level: int = 3) -> None:
        """board: the workspace whose mirror scope, ignore rules and onboarding records are read.

        level: the zstd level the measurement compresses at.
        """
        self.board = board
        self.level = level
        self.ignore = GitignoreFilter(board.root)
        self.nested: dict[Path, pathspec.GitIgnoreSpec] = {}

    @property
    def root(self) -> Path:
        """The workspace root, the board's own."""
        return self.board.root

    @staticmethod
    def denylist(excluded: Sequence[str]) -> pathspec.GitIgnoreSpec:
        """What the mirror refuses to carry: its permanent denylist and this host's own excludes.

        The two pattern languages agree on wildcards and disagree on one thing that matters here.
        A pattern carrying a slash is anchored to the root for git and matches at any depth for
        rsync, so `data/raw` excludes every `data/raw` in the tree when rsync reads it. Anchoring
        it with `**/` restores rsync's own reading, which is the one the mirror will apply.

        excluded: the host profile's declared exclude patterns.
        """
        patterns = [Transfer._anywhere(pattern) for pattern in (*sync.ALWAYS_EXCLUDE, *excluded)]
        # pyrefly: ignore  reason=pathspec from_lines stub over-narrows to AnyStr since=2026-08-16
        return pathspec.GitIgnoreSpec.from_lines(patterns)

    def compressed(self, files: Sequence[Path]) -> tuple[int, int]:
        """The raw and compressed size of `files` read through one zstd stream."""
        compressor = zstd.ZstdCompressor(level=self.level)
        raw = wire = 0
        for path in files:
            with path.open("rb") as opened:
                while chunk := opened.read(_CHUNK):
                    raw += len(chunk)
                    wire += len(compressor.compress(chunk))
        return raw, wire + len(compressor.flush())

    def descend(self, directory: Path, denied: pathspec.GitIgnoreSpec) -> Iterator[Path]:
        """Every mirrorable file under `directory`, an excluded subtree never entered at all.

        Pruning rather than filtering, because the trees the mirror refuses are exactly the big
        ones, a Rust `target/` or a virtual environment, and walking one only to drop every file
        it holds is the difference between a measurement that takes a second and one that takes
        a minute.
        """
        for entry in sorted(directory.iterdir()):
            relative = entry.relative_to(self.root)
            folder = entry.is_dir()
            if denied.match_file(f"{relative}/" if folder else str(relative)):
                continue
            if self.ignored(relative, folder=folder):
                continue
            if folder and not entry.is_symlink():
                yield from self.descend(entry, denied)
            elif entry.is_file():
                yield entry

    def ignored(self, relative: Path, *, folder: bool) -> bool:
        """Whether git ignores `relative`, under the workspace's rules and every nested one.

        The mirror hands rsync a per-directory merge rule, so a `.gitignore` deep in the tree
        prunes its own subtree exactly as it does for git, and a package that excludes its own
        build output is honoured without the workspace root having to know about it. Reading
        only the root file here would count that build output as in flight.

        relative: the path being judged, relative to the workspace root.
        folder: whether it is a directory, which is what a `build/` rule matches on.
        """
        return self.ignore.ignored(relative) or any(
            self.pruned(parent, relative=relative, folder=folder)
            for parent in relative.parents
            if parent != Path()
        )

    def measure(self, files: Sequence[Path]) -> tuple[int, int]:
        """The raw and compressed size of `files`, over a small pool of parallel zstd streams.

        zstd releases the interpreter lock while it compresses, so this is one of the few places
        in this package where threads buy real time on the plain build rather than only on the
        free-threaded one: 2.76x at eight threads over 600 MB of this workspace's own files,
        against 1.57x at two and 2.66x at sixteen.

        Each shard is its own stream, which is also the shape the mirror sends, since rsync
        compresses what it ships rather than handing the whole set to one long-lived context.
        """
        shards = [shard for index in range(_STREAMS) if (shard := files[index::_STREAMS])]
        if not shards:
            return 0, 0
        with ThreadPoolExecutor(max_workers=len(shards)) as pool:
            measured = list(pool.map(self.compressed, shards))
        return sum(raw for raw, _ in measured), sum(wire for _, wire in measured)

    def newer(self, path: Path, since: str) -> bool:
        """Whether `path` changed after the `since` watermark, true when there is no watermark."""
        return not since or path.stat().st_mtime > datetime.fromisoformat(since).timestamp()

    def pruned(self, parent: Path, *, relative: Path, folder: bool) -> bool:
        """Whether `parent`'s own `.gitignore` prunes `relative`, false when it declares none.

        Almost no directory in a real tree declares one, and an empty rule set still charges for
        the path arithmetic and the match before answering no. Every file asks every ancestor, so
        that empty work is most of the walk, and stepping over it is worth 1.3x on this
        workspace's tree with a byte-identical file set.

        parent: an ancestor directory of `relative`, relative to the workspace root.
        relative: the path being judged, relative to the workspace root.
        folder: whether it is a directory, which is what a `build/` rule matches on.
        """
        rules = self.rules(parent)
        if not rules.patterns:
            return False
        inside = relative.relative_to(parent)
        return rules.match_file(f"{inside}/" if folder else str(inside))

    def rules(self, directory: Path) -> pathspec.GitIgnoreSpec:
        """The `.gitignore` rules `directory` declares, empty when it declares none.

        Cached per directory, since a deep walk asks the same handful of directories about every
        file beneath them.
        """
        if directory not in self.nested:
            gitignore = self.root / directory / ".gitignore"
            lines = (
                gitignore.read_text(encoding="utf-8").splitlines() if gitignore.is_file() else []
            )
            # pyrefly: ignore  reason=pathspec from_lines stub over-narrows to AnyStr since=2026-08-16
            self.nested[directory] = pathspec.GitIgnoreSpec.from_lines(lines)
        return self.nested[directory]

    def set_for(self, job: BatchJob) -> TransferSet:
        """What `job` ships to its target, measured now.

        A job on this machine ships nothing, since the workspace is already here. Everything
        else is the mirror's delta plus the job's own named data, measured compressed because
        that is how it crosses the wire.
        """
        if self.board.on(job.target).local:
            return TransferSet(job=job.name, target=job.target)
        since = self.watermark(job.target)
        scope = self.board.on(job.target).plan().profile.sync
        denied = self.denylist(scope.exclude)
        changed = [path for path in self.walk(scope.include, denied) if self.newer(path, since)]
        named = list(self.walk(job.data, denied))
        files = list(dict.fromkeys([*changed, *named]))
        raw, wire = self.measure(files)
        return TransferSet(
            job=job.name,
            target=job.target,
            paths=(*scope.include, *job.data),
            files=len(files),
            raw_bytes=raw,
            wire_bytes=wire,
            since=since,
        )

    def walk(self, paths: Sequence[str], denied: pathspec.GitIgnoreSpec) -> Iterator[Path]:
        """Every mirrorable file under each of `paths`, in a stable order.

        A declared path that does not exist here is skipped the way the mirror skips it, since a
        stale include line is a warning at dispatch rather than a refusal.

        paths: the declared roots, each a file or a directory under the workspace.
        denied: the profile's own exclude patterns and the mirror's permanent denylist.
        """
        for declared in paths:
            start = self.root / declared
            if start.is_file():
                yield start
            elif start.is_dir():
                yield from self.descend(start, denied)

    @staticmethod
    def _anywhere(pattern: str) -> str:
        """`pattern` as rsync reads it: anchored only when it was written with a leading slash."""
        unanchored = "/" in pattern.rstrip("/") and not pattern.startswith("/")
        return f"**/{pattern}" if unanchored else pattern

    def watermark(self, alias: str) -> str:
        """When `alias` last had the workspace mirrored onto it, empty when nothing recorded one.

        Empty is not "never mirrored", it is "this workspace has no record of a mirror", and the
        two are the same thing to a transfer set: with nothing to subtract, everything the scope
        names is in flight.
        """
        try:
            return self.board.dispatcher.cache.host(alias).mirrored_at
        except LookupError:
            return ""
