from collections import deque
from collections.abc import Callable

from patos import Model
from pydantic import Field

from ...models import ProcessReading


class SpanFrame(Model):
    """One live span with a bounded, mutable telemetry buffer."""

    name: str
    path: str
    thread: int
    device_start_ns: int
    finish_marker: Callable[[], None] | None
    samples: deque[ProcessReading] = Field(default_factory=lambda: deque(maxlen=4096))
