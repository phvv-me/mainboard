# What a job still has to put on its target, and what that costs on the wire.
#
# An onboarded host already carries the workspace, so a job ships the files changed since its
# mirror was last refreshed, plus the data the job names, which the mirror's scope may not carry.

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

# Read size while measuring: big enough to be disk-bound, small enough never to hold a whole file.
_CHUNK = 1 << 20

# Parallel zstd streams per measurement. zstd releases the GIL, and over 600 MB of this workspace
# on eight cores it gave 1.57x at two threads, 2.35x at four, 2.76x at eight, 2.66x at sixteen.
_STREAMS = 8


class TransferSet(FrozenModel):
    """What one job still has to ship, measured rather than guessed.

    paths: the roots measured (the mirror's include scope and the job's data), so a surprising
        size traces back to what was counted.
    wire_bytes: the compressed size every transfer has, estimated over parallel zstd streams.
    since: the mirror watermark the delta was measured against, empty when the target has no
        recorded mirror and everything in scope is in flight.
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

    Every path is walked through the very scope the dispatch mirrors (include list, ignore rules,
    host excludes, generated directories), so what is measured is what would ship.
    """

    def __init__(self, board: Board, *, level: int = 3) -> None:
        self.board = board
        self.level = level

    @property
    def root(self) -> Path:
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
        """The raw and compressed size of `files` over `_STREAMS` parallel zstd streams, an
        estimate of the mirror's one stream rather than a replay."""
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
        """What `job` ships to its target, measured now; nothing for this machine."""
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

        A missing path is skipped as the mirror skips it (a stale include is a warning at dispatch,
        not a refusal), and a link is never followed into a second copy of the tree.
        """
        scope = self.board.dispatcher.scope(plan, paths)
        return [
            self.root / entry.path for entry in walk(str(self.root), scope) if entry.kind == FILE
        ]

    def watermark(self, alias: str) -> str:
        """When `alias` last had the workspace mirrored onto it, empty with no record of a mirror,
        leaving everything in scope in flight."""
        try:
            return self.board.dispatcher.cache.host(alias).mirrored_at
        except LookupError:
            return ""
