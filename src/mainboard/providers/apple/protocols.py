"""Structural contracts for the untyped Apple `os_signpost` and `system_profiler` surfaces.

The `os-signpost` package ships no stubs, so these Protocols pin the `Signposter` interval
API the tracer drives and the opaque interval token it threads from begin to end.
"""

from typing import Protocol


class IntervalToken(Protocol):
    """An opaque `os_signpost` interval handle, only paired back into `end_interval`."""


class Signposter(Protocol):
    """The `os_signpost.Signposter` interval surface mainboard emits to Instruments."""

    def begin_interval(self, name: str) -> IntervalToken: ...
    def end_interval(self, name: str, token: IntervalToken) -> None: ...
    def emit_event(self, name: str) -> None: ...


class SignpostModule(Protocol):
    """The `os_signpost` module: its `Signposter` factory bound to a subsystem string."""

    def Signposter(self, subsystem: str) -> Signposter: ...  # noqa: N802 — the binding's class
