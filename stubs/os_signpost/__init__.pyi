# Stub for `os_signpost` (0.0.3, macOS only, no types shipped): intervals to Instruments.

import enum
from collections.abc import Callable

class Signposter:
    class Category(enum.Enum):
        PointsOfInterest = "PointsOfInterest"

    def __init__(self, subsystem: str, category: Signposter.Category | str) -> None: ...
    def begin_interval(self, msg: str) -> Callable[[str | None], None]: ...
    def emit_event(self, msg: str) -> None: ...
