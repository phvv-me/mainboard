"""The `collect` pass: mb's spooled events sent to the sink, which stays the lake's one writer.

Every minute it reads each spool file from where it last stopped, POSTs each event as its own
request (the sink refuses a body of two lines), and keeps the byte offset it delivered up to in
`collected.json`. The sink stores a line twice if it is sent twice, so a delivery is kept single
twice over: by the offset, kept once a file's pending lines are all accepted, and by the event's
id, which a pass first looks for in the lake, so a line a pass cut short had sent is not sent
again once the sink has stored it. One pass delivers at a time, the collect or the drain, under
a lock beside the ledger. A file delivered whole that no writer holds is deleted; one holding a
line that is not a spool line is set aside as `.refused`, never retried, since skipping the line
would hide the bug that wrote it.
"""

import http.client
from contextlib import closing
from functools import partial
from typing import TYPE_CHECKING

from filelock import FileLock, Timeout, lock_descriptor

from ...core.errors import MissionError
from ...core.events.spool import EventSpool
from ...log import logger

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence, Set
    from pathlib import Path
    from uuid import UUID

    from ...core.events.spool import Page
    from .sink import Sink

# How long a pass waits for a sink that is starting beside it, and how long one POST may take.
_STARTING = 15.0
_POSTING = 10.0
_ACCEPTED = range(200, 300)
_LOCK = "collect.lock"


class Collector:
    """The collect pass of one workspace's spool into its sink."""

    @staticmethod
    def _delivered(
        path: Path,
        page: Page,
        *,
        held: Set[UUID],
        offsets: dict[str, int],
        post: Callable[[str], None],
    ) -> int:
        """Send `page`'s events the lake does not hold and keep `path`'s new offset; how many."""
        sending = [event for event in page.events if event.id not in held]
        for event in sending:
            post(event.line())
        offsets[path.name] = page.offset
        if page.offset == path.stat().st_size and Collector._retired(path):
            del offsets[path.name]
        return len(sending)

    @staticmethod
    def _post(
        connection: http.client.HTTPConnection, headers: Mapping[str, str], line: str
    ) -> None:
        """Send one spool line, refused unless the sink accepts it."""
        connection.request("POST", "/v1/logs", line.encode(), dict(headers))
        answer = connection.getresponse()
        said = answer.read()[:200].decode(errors="replace")
        if answer.status not in _ACCEPTED:
            raise MissionError(f"the sink answered {answer.status} to a spool line: {said}")

    @staticmethod
    def _refused(path: Path, fault: ValueError, offsets: dict[str, int]) -> None:
        """Set `path` aside as `.refused`, never to be read again, and forget its offset."""
        logger.warning(f"set aside, never retried: {fault}")
        path.rename(path.with_suffix(".refused"))
        offsets.pop(path.name, None)

    @staticmethod
    def _retired(path: Path) -> bool:
        """Delete `path` unless a writer holds it, under the lock every spool writer keeps."""
        with path.open("rb") as handle:
            if not lock_descriptor(handle.fileno(), blocking=False):
                return False
            path.unlink()
            return True

    def __init__(self, sink: Sink) -> None:
        self.sink = sink

    def _page(self, path: Path, offsets: dict[str, int]) -> Page | None:
        """The events of `path` past its offset; None once it is set aside for a bad line."""
        try:
            return EventSpool.read(path, offsets.get(path.name, 0))
        except ValueError as fault:
            Collector._refused(path, fault, offsets)
            return None

    def _posted(self, files: Sequence[Path], offsets: dict[str, int]) -> int:
        """Send what `files` hold past `offsets` but the events the lake holds; how many."""
        pages = {path: page for path in files if (page := self._page(path, offsets)) is not None}
        held = self.sink.lake.held([event for page in pages.values() for event in page.events])
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.sink.token.read()}",
        }
        with closing(self.sink.connection(_POSTING)) as connection:
            post = partial(Collector._post, connection, headers)
            return sum(
                Collector._delivered(path, page, held=held, offsets=offsets, post=post)
                for path, page in pages.items()
            )

    def _sent(self, files: Sequence[Path]) -> int:
        """`_posted` over the ledger's offsets, which it keeps however the sending ends."""
        offsets = self.sink.lake.offsets()
        try:
            return self._posted(files, offsets)
        finally:
            self.sink.lake.keep(offsets)

    def collect(self) -> int:
        """Deliver every spooled event the lake does not yet hold; answers how many were sent.

        Refused when the sink does not answer within `_STARTING`, every line left waiting; sends
        nothing while another pass delivers.
        """
        files = sorted(self.sink.lake.spool.glob("*.jsonl"))
        if not files:
            return 0
        if not self.sink.awaited(_STARTING):
            raise MissionError(
                f"the sink does not answer at {self.sink.endpoint}; {len(files)} spool files wait"
            )
        try:
            held = FileLock(self.sink.lake.home / _LOCK, timeout=0, mode=0o600).acquire()
        except Timeout:
            logger.info("another pass is delivering the spool, so this one sends nothing")
            return 0
        with held:
            return self._sent(files)
