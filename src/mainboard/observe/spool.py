# The node-side half of live observability: append frames to a job's own directory, roll and
# compress a segment once it grows past a threshold, and publish a small heartbeat a poll
# channel can cat. Everything here is stdlib only, since this is the code a bare remote host
# runs with no guarantee any dependency beyond Python itself is installed.

import json
from compression import zstd
from datetime import UTC, datetime
from time import sleep
from typing import TYPE_CHECKING

from .frames import Frame, Kind, encode, next_offset, parse_tail

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from pathlib import Path
    from types import TracebackType

    from .frames import JSONValue

_ROLL_BYTES = 4 * 1024 * 1024

_LIVE_NAME = "live.ndjson"
_STATUS_NAME = "status.json"


class Spool:
    """One job's append-only NDJSON stream on disk, rolled and compressed as it grows.

    A fresh instance recovers its write position from whatever is already on disk, so
    reopening the same `(root, job)` after a restart resumes rather than overwriting.
    """

    def __init__(self, root: Path, job: str, *, roll_bytes: int = _ROLL_BYTES) -> None:
        self.dir = root / job
        self.dir.mkdir(parents=True, exist_ok=True)
        self.job = job
        self.roll_bytes = roll_bytes
        self.segment_start, self.offset = self.__resume()
        self.handle = self.__live_path().open("ab")

    def __enter__(self) -> Spool:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc_value, traceback
        self.close()

    def append(self, frame: Frame) -> Frame:
        """Write `frame` at the current stream position and roll the segment if it grew enough.

        frame: the event to record; its `offset` is overwritten with the true write position.
        """
        stamped = frame.model_copy(update={"offset": self.offset})
        body = encode(stamped).encode()
        self.handle.write(body)
        self.handle.flush()
        self.offset += len(body)
        if self.offset - self.segment_start >= self.roll_bytes:
            self.__roll()
        return stamped

    def close(self) -> None:
        """Release the live segment's file handle."""
        self.handle.close()

    def frames_from(self, offset: int) -> list[Frame]:
        """Every frame at or after the global `offset`, spanning archived and live segments.

        offset: the resume checkpoint a previous fetch or heartbeat reported.
        """
        chunks: list[bytes] = []
        for start, end, path, compressed in self.__segments():
            if end <= offset:
                continue
            raw = path.read_bytes()
            data = zstd.decompress(raw) if compressed else raw
            chunks.append(data[max(0, offset - start) :])
        return parse_tail(b"".join(chunks).decode())

    def heartbeat(self, state: str) -> None:
        """Atomically publish `status.json`, the small summary a poll channel cats.

        state: a free-form label for the job's current lifecycle stage (`running`, `ended`).
        """
        payload = {
            "state": state,
            "offset": self.offset,
            "updated_at": datetime.now(UTC).isoformat(),
            "base": self.segment_start,
        }
        tmp = self.dir / f"{_STATUS_NAME}.tmp"
        tmp.write_text(json.dumps(payload))
        tmp.replace(self.dir / _STATUS_NAME)

    def status(self) -> dict[str, JSONValue] | None:
        """The last published heartbeat, or `None` before any heartbeat has landed."""
        try:
            return json.loads((self.dir / _STATUS_NAME).read_text())
        except FileNotFoundError:
            return None

    def __archives(self) -> list[tuple[int, int, Path, bool]]:
        """Every already-rolled, compressed segment, oldest first, from its own filename."""
        entries = []
        for path in sorted(self.dir.glob("*.ndjson.zst")):
            start, end = (int(part) for part in path.name.removesuffix(".ndjson.zst").split("-"))
            entries.append((start, end, path, True))
        return entries

    def __live_path(self) -> Path:
        return self.dir / _LIVE_NAME

    def __resume(self) -> tuple[int, int]:
        """The live segment's start and current write position, recovered from disk."""
        segments = self.__segments()
        if not segments:
            return 0, 0
        start, end, _, compressed = segments[-1]
        return (end, end) if compressed else (start, end)

    def __roll(self) -> None:
        """Compress the live segment into an archive and start a fresh empty one."""
        self.handle.close()
        live = self.__live_path()
        data = live.read_bytes()
        end = self.segment_start + len(data)
        archive = self.dir / f"{self.segment_start:020d}-{end:020d}.ndjson.zst"
        archive.write_bytes(zstd.compress(data))
        live.unlink()
        self.segment_start = end
        self.handle = live.open("ab")

    def __segments(self) -> list[tuple[int, int, Path, bool]]:
        """Every segment (archived, then live if present) as `(start, end, path, compressed)`."""
        archives = self.__archives()
        live = self.__live_path()
        if not live.exists():
            return archives
        base = archives[-1][1] if archives else 0
        return [*archives, (base, base + live.stat().st_size, live, False)]


def follow(
    spool: Spool,
    offset: int,
    *,
    interval: float = 0.5,
    sleeper: Callable[[float], None] = sleep,
) -> Iterator[Frame]:
    """Replay `spool` from `offset`, then keep tailing until the job's `ended` frame lands.

    offset: the checkpoint to replay from, typically the caller's last-seen position.
    interval: real seconds paused between polls of a still-running job.
    sleeper: overrides the pacing for a hermetic test; unset means sleep for real.
    """
    cursor = offset
    while True:
        frames = spool.frames_from(cursor)
        yield from frames
        cursor = next_offset(frames, default=cursor)
        if any(frame.kind is Kind.ended for frame in frames):
            return
        status = spool.status()
        if not frames and status is not None and status.get("state") == "ended":
            return
        sleeper(interval)
