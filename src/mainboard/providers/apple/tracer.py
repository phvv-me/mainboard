"""macOS annotation backend: `os_signpost` intervals, shown in Instruments.

Ranges land in the Points of Interest track and are near-free when no Instruments
session is recording. Uses the `os-signpost` package; absent it, this backend is
simply unavailable and the no-op base is used instead.
"""

import platform
from contextlib import suppress
from importlib import import_module
from typing import TYPE_CHECKING, ClassVar, cast

from ...enums import Vendor
from ...profiling.tracer import Marker, Tracer

if TYPE_CHECKING:
    from .protocols import IntervalToken, SignpostModule


def _load_signpost() -> SignpostModule:
    return cast("SignpostModule", import_module("os_signpost"))


_signpost: SignpostModule | None = None
with suppress(ImportError):
    _signpost = _load_signpost()

_SUBSYSTEM = "me.phvv.mainboard"


class SignpostTracer(Tracer):
    """`os_signpost` intervals/events; keeps a (name, token) stack for push/pop."""

    vendor: ClassVar[Vendor] = Vendor.APPLE
    label: ClassVar[str] = "signpost"

    def __init__(self) -> None:
        assert _signpost is not None  # SignpostTracer is built only when the module loaded
        self._signposter = _signpost.Signposter(_SUBSYSTEM)
        self._stack: list[tuple[str, IntervalToken]] = []

    @classmethod
    def is_available(cls) -> bool:
        return _signpost is not None and platform.system() == "Darwin"

    def push(self, name: str) -> None:
        self._stack.append((name, self._signposter.begin_interval(name)))

    def pop(self) -> None:
        if self._stack:
            name, token = self._stack.pop()
            self._signposter.end_interval(name, token)

    def start(self, name: str) -> Marker:
        """Open a correlatable signpost interval and return its exact closer."""
        token = self._signposter.begin_interval(name)
        return lambda: self._signposter.end_interval(name, token)

    def mark(self, name: str) -> None:
        self._signposter.emit_event(name)
