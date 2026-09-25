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

from patos import FrozenModel

from ..dispatch.agent import walk
from ..dispatch.agent.program import FILE

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..board import Board
    from ..context.plan import ExecutionPlan
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
    wire_bytes: their size compressed, since every transfer this tool makes is compressed.
        Measured over a few parallel zstd streams, an estimate of the mirror's own stream that
        costs a fraction of the time one long-lived context over the whole set would.
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

    Every path is walked through the very scope the dispatch mirrors, so what is measured here
    is what would actually ship: the host profile's include list, the workspace's own ignore
    rules, the host's excludes and the generated directories the mirror never carries.
    """

    def __init__(self, board: Board, *, level: int = 3) -> None:
        """board: the workspace whose mirror scope, ignore rules and onboarding records are read.

        level: the zstd level the measurement compresses at.
        """
        self.board = board
        self.level = level

    @property
    def root(self) -> Path:
        """The workspace root, the board's own."""
        return self.board.root

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

    def measure(self, files: Sequence[Path]) -> tuple[int, int]:
        """The raw and compressed size of `files`, over a small pool of parallel zstd streams.

        zstd releases the interpreter lock while it compresses, so this is one of the few places
        in this package where threads buy real time on the plain build rather than only on the
        free-threaded one: 2.76x at eight threads over 600 MB of this workspace's own files,
        against 1.57x at two and 2.66x at sixteen.

        Each shard is its own stream, so the measurement parallelizes where the mirror's one
        stream would not, and stays an estimate of it rather than a replay.
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

    def set_for(self, job: BatchJob) -> TransferSet:
        """What `job` ships to its target, measured now.

        A job on this machine ships nothing, since the workspace is already here. Everything
        else is the mirror's delta plus the job's own named data, measured compressed because
        that is how it crosses the wire.
        """
        if self.board.on(job.target).local:
            return TransferSet(job=job.name, target=job.target)
        since = self.watermark(job.target)
        plan = self.board.on(job.target).plan()
        scope = plan.profile.sync
        changed = [path for path in self.walk(plan, scope.include) if self.newer(path, since)]
        named = self.walk(plan, job.data)
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

    def walk(self, plan: ExecutionPlan, paths: Sequence[str]) -> list[Path]:
        """Every file the mirror would carry under each of `paths`, in a stable order.

        A declared path that does not exist here is skipped the way the mirror skips it, since a
        stale include line is a warning at dispatch rather than a refusal, and a link is never
        followed into a second copy of the tree.

        plan: the target's resolved execution context, whose profile names its excludes.
        paths: the declared roots, each a file or a directory under the workspace.
        """
        scope = self.board.dispatcher.scope(plan, paths)
        return [
            self.root / entry.path for entry in walk(str(self.root), scope) if entry.kind == FILE
        ]

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
