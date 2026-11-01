# The host-side half of a job's frames in OpenTelemetry's shape: the runner starts, beats and
# settles a spool file that sits beside the job's output, and the center reads it when the run
# settles. Frames are telemetry, lossy and untrusted by rule: a spool that cannot be opened costs
# the job nothing, and a job killed outright leaves a start and its beats but no settle.

import threading
from collections.abc import Callable, Generator, Mapping
from contextlib import contextmanager, suppress
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic
from uuid import uuid7

import psutil
from patos import FrozenModel

from ...constants import FRAMES_MARK
from ...enums import JobFrame, JobReading, Stop
from ...log import logger
from ..errors import MissionError
from .identity import Attribute, Identity
from .spool import Event, EventSpool

type Reading = int | float | str


class FrameSpec(FrozenModel):
    """Where a job writes its frames and as which host, part of the record the host runs.

    directory: the host folder the job's spool file goes in, under the mirror's state directory.
    host: the mb alias of the machine the job runs on.
    """

    directory: str
    host: str


class JobFrames:
    """The emitter one job's runner calls: a frame at the start, every beat, and the settle.

    spec: where to write; None emits nothing.
    say: how the job's own output names its spool file, which is how the center finds it.
    interval: seconds between beats.
    """

    @contextmanager
    def running(self, pid: Callable[[], int | None]) -> Generator[None]:
        """Emit the start, beat while the body runs, and let go of the spool however it ends.

        pid: the command's process once it has started, else None.
        """
        self._start()
        self._emit(JobFrame.started)
        beating = threading.Thread(target=self._beat, args=(pid,), daemon=True)
        beating.start()
        try:
            yield
        finally:
            self._stopped(beating)

    @staticmethod
    def _cpu(member: psutil.Process) -> float:
        with suppress(psutil.NoSuchProcess, psutil.AccessDenied):
            times = member.cpu_times()
            return times.user + times.system
        return 0.0

    @staticmethod
    def _rss(member: psutil.Process) -> int:
        with suppress(psutil.NoSuchProcess, psutil.AccessDenied):
            return member.memory_info().rss
        return 0

    def __init__(
        self, spec: FrameSpec | None, *, say: Callable[[str], None], interval: float = 60.0
    ) -> None:
        self.spec = spec
        self.say = say
        self.interval = interval
        self.spool: EventSpool | None = None
        self.lock = threading.Lock()
        self.over = threading.Event()
        self.begun = monotonic()

    def _beat(self, pid: Callable[[], int | None]) -> None:
        """Emit a heartbeat every `interval` seconds until the job settles."""
        while not self.over.wait(self.interval):
            self._emit(JobFrame.heartbeat, self._counters(pid()))

    def _counters(self, pid: int | None) -> dict[JobReading, Reading]:
        """What the command's tree uses now; only the clock before it starts or once it is gone."""
        counters: dict[JobReading, Reading] = {JobReading.wall_s: self._wall()}
        if pid is None:
            return counters
        with suppress(psutil.NoSuchProcess):
            root = psutil.Process(pid)
            tree = [root, *root.children(recursive=True)]
            counters |= {
                JobReading.processes: len(tree),
                JobReading.rss_bytes: sum(self._rss(member) for member in tree),
                JobReading.cpu_s: round(sum(self._cpu(member) for member in tree), 2),
            }
        return counters

    def _emit(self, frame: JobFrame, readings: Mapping[JobReading, Reading] | None = None) -> None:
        """Append `frame` unless no spool could be opened or the job has settled."""
        identity = Identity(host=self.spec.host) if self.spec else Identity()
        attributes: dict[str, Attribute] = {
            key.value: v for key, v in identity.attributes().items()
        }
        attributes |= {key.value: value for key, value in (readings or {}).items()}
        with self.lock:
            if self.spool is None or self.over.is_set():
                return
            self.spool.append(Event(name=frame, at=datetime.now(UTC), attributes=attributes))
            if frame is JobFrame.settled:
                self.over.set()

    def _start(self) -> None:
        """Start the clock and open a spool file named by its own id, said in the job's output.

        A spool that cannot be opened is said in the log and leaves the job without frames.
        """
        self.begun = monotonic()
        if self.spec is None:
            return
        name = f"{uuid7().hex}.jsonl"
        try:
            self.spool = EventSpool(Path(self.spec.directory) / name)
        except (OSError, MissionError) as fault:
            logger.warning("no frames for this job: {}", fault)
            return
        self.say(f"{FRAMES_MARK}{name}")

    def _stopped(self, beating: threading.Thread) -> None:
        """Stop the beats and let go of the spool."""
        self.over.set()
        beating.join()
        if self.spool is not None:
            self.spool.close()

    def _wall(self) -> float:
        return round(monotonic() - self.begun, 1)

    def settled(self, exit_code: int, stop: Stop) -> None:
        """Emit the last frame, how the job ended; nothing follows it."""
        readings: dict[JobReading, Reading] = {
            JobReading.wall_s: self._wall(),
            JobReading.exit_code: exit_code,
            JobReading.stop: stop.value,
        }
        self._emit(JobFrame.settled, readings)
