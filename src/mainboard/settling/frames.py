"""A settled run's frames, moved from its host into the telemetry spool for the collect pass.

The runner names its spool file in the job's output (`FRAMES_MARK`), and settling already keeps
that output, so the file is found without asking the scheduler anything it cannot answer. The
job's own command can write to that file and print that line too, so a frame is kept only as the
three events the runner emits, with identities from the dispatch record rather than the file,
numeric counters, and a stop reason from `Stop`: nothing else the file holds reaches the spool.

Nothing is read or spooled before the sink has made the telemetry lake, so a workspace that never
runs `mb serve` pays no round trip for it. The spool file is named for the run and appends only
the events it does not hold; the lake's own check of event ids keeps a file re-spooled after the
collect pass deleted it from being stored twice.
"""

import re
import shlex
from typing import TYPE_CHECKING

from ..constants import FRAMES_MARK
from ..core.errors import MissionError
from ..core.events import Event, EventSpool, Key
from ..dispatch.schedulers import HostUnreachable
from ..dispatch.shared import state_dir
from ..dispatch.wrapping import connection
from ..enums import JobFrame, JobReading, Stop
from ..log import logger

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from ..core.events.identity import Attribute
    from ..dispatch.dispatcher import Handle
    from ..state.telemetry import TelemetryLake

# The most bytes read from a host's file, about a thousand frames.
_MOST = 1 << 20
_NAME = re.compile(rf"^{re.escape(FRAMES_MARK)}([0-9a-f]{{32}}\.jsonl)$", re.MULTILINE)
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")
_COUNTERS = frozenset(JobReading) - {JobReading.stop}
_FRAMES = frozenset(JobFrame)


class FrameRelay:
    """Moves the frames of settled runs from their hosts into one workspace's telemetry spool."""

    @staticmethod
    def _kept(attributes: Mapping[str, Attribute]) -> dict[str, Attribute]:
        """The readings of a frame that are numbers, and a stop reason that is one of ours."""
        return {
            name: value
            for name, value in attributes.items()
            if (name in _COUNTERS and type(value) in {int, float})
            or (name == JobReading.stop and value in tuple(Stop))
        }

    def __init__(self, lake: TelemetryLake) -> None:
        self.lake = lake

    def _events(self, text: str, *, handle: Handle) -> list[Event]:
        """The frames in `text`, with the identities of `handle`'s run and only their readings."""
        identity: dict[str, Attribute] = {
            Key.host.value: handle.host,
            Key.job.value: f"{handle.host}/{handle.id}",
        }
        events: list[Event] = []
        for line in filter(None, text.rpartition("\n")[0].split("\n")):
            try:
                event = Event.parse(line)
            except ValueError as fault:
                logger.warning("skipped a frame of {}: {}", handle.id, fault)
                continue
            if event.name in _FRAMES:
                attributes = {**identity, **self._kept(event.attributes)}
                events.append(event.model_copy(update={"attributes": attributes, "body": ""}))
        return events

    def _read(self, handle: Handle, name: str) -> str:
        """The host file `name` of the run's mirror, as far as `_MOST` bytes of it go."""
        path = shlex.quote(f"{handle.root}/{state_dir()}/frames/{name}")
        with connection(handle.host) as remote:
            return remote["bash"][["-lc", f"head -c {_MOST} {path} 2>/dev/null"]](retcode=None)

    def _relayed(self, handle: Handle, names: Sequence[str]) -> int:
        """Spool what the host's files `names` hold that the run's spool does not; how many."""
        return sum(
            self._spooled(self._events(self._read(handle, name), handle=handle), handle=handle)
            for name in names
        )

    def _spooled(self, events: Sequence[Event], *, handle: Handle) -> int:
        """Append the events the run's spool file does not hold yet; how many that was."""
        path = self.lake.spool / f"jobs-{_UNSAFE.sub('-', f'{handle.host}-{handle.id}')}.jsonl"
        with EventSpool(path) as spool:
            held = {event.id for event in EventSpool.read(path).events}
            fresh = [event for event in events if event.id not in held]
            for event in fresh:
                spool.append(event)
        return len(fresh)

    def pull(self, handle: Handle, transcript: str) -> int:
        """Spool the frames `transcript` names that the run's host holds; how many were new.

        A host that does not answer costs the run nothing, since frames are telemetry.
        """
        names = list(dict.fromkeys(_NAME.findall(transcript)))
        if not (names and handle.root and self.lake.attachment.exists()):
            return 0
        try:
            return self._relayed(handle, names)
        except (HostUnreachable, MissionError, OSError) as fault:
            logger.warning("no frames from {} for {}: {}", handle.host, handle.id, fault)
            return 0
