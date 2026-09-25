# Structural contracts for the untyped `os-signpost` package the tracer drives.

from collections.abc import Callable
from typing import Protocol


class IntervalToken(Protocol):
    """An opaque `os_signpost` interval handle, only paired back into `end_interval`."""


class Signposter(Protocol):
    """The `os_signpost.Signposter` interval surface the tracer emits to Instruments."""

    def begin_interval(self, name: str) -> IntervalToken: ...
    def emit_event(self, name: str) -> None: ...
    def end_interval(self, name: str, token: IntervalToken) -> None: ...


class SignpostModule(Protocol):
    """The `os_signpost` module: its `Signposter` factory bound to a subsystem string."""

    Signposter: Callable[[str], Signposter]
