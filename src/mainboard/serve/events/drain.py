"""The `drain` pass: what still waits in mb's spool, delivered to the sink once every five minutes.

The collect pass delivers each minute and fails its run while the sink does not answer, which is
how the supervisor shows a sink down. The drain is its backstop and stays quiet: it reads the
lake's `pending` view first and does nothing when nothing waits, keeps every line and ends clean
while the sink is down, and otherwise delivers through the collect pass's own path, which sends
no event whose id the lake already holds, so draining twice stores each event once. Asks are
spool lines like any other, so the drain also brings them to the lake that `serve asks` reads.
"""

from typing import TYPE_CHECKING

from ...log import logger
from .collect import Collector

if TYPE_CHECKING:
    from .sink import Sink


class Drain:
    """The drain pass of one workspace's spool into its sink."""

    def __init__(self, sink: Sink) -> None:
        self.sink = sink

    def drain(self) -> int:
        """Deliver what waits, once; answers how many events were sent, none with the sink down.

        A file set aside for a line that is no spool line is never sent, so it does not count.
        """
        waiting = [file for file in self.sink.lake.pending() if file.oldest is not None]
        since = [file.oldest for file in waiting if file.oldest is not None]
        if not since:
            return 0
        if self.sink.answering():
            try:
                return Collector(self.sink).collect()
            except (ConnectionError, TimeoutError) as fault:
                logger.warning(f"the sink stopped answering mid-pass, what it lost waits: {fault}")
                return 0
        lines = sum(file.lines for file in waiting)
        logger.warning(
            f"the sink does not answer at {self.sink.endpoint}, so {lines} events in "
            f"{len(waiting)} spool files wait, the oldest from {min(since).astimezone():%c}"
        )
        return 0
