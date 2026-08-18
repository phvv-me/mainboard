# macOS annotation backend: `os_signpost` intervals, shown in Instruments.

import platform
from contextlib import suppress
from importlib import import_module
from typing import TYPE_CHECKING, ClassVar, cast

from ...tracer import Marker, Tracer, Vendor

if TYPE_CHECKING:
    from .protocols import IntervalToken, SignpostModule

_signpost: SignpostModule | None = None
with suppress(ImportError):
    _signpost = cast("SignpostModule", import_module("os_signpost"))

_SUBSYSTEM = "me.phvv.mainboard"


class SignpostTracer(Tracer):
    """`os_signpost` intervals/events; keeps a (name, token) stack for push/pop."""

    vendor: ClassVar[Vendor] = Vendor.APPLE
    label: ClassVar[str] = "signpost"

    def __init__(self) -> None:
        assert _signpost is not None  # noqa: S101  reason=built only when signpost loaded since=2026-08-16
        self._signposter = _signpost.Signposter(_SUBSYSTEM)
        self._stack: list[tuple[str, IntervalToken]] = []

    @classmethod
    def is_available(cls) -> bool:
        return _signpost is not None and platform.system() == "Darwin"

    def mark(self, name: str) -> None:
        self._signposter.emit_event(name)

    def pop(self) -> None:
        if self._stack:
            name, token = self._stack.pop()
            self._signposter.end_interval(name, token)

    def push(self, name: str) -> None:
        self._stack.append((name, self._signposter.begin_interval(name)))

    def start(self, name: str) -> Marker:
        """Open a correlatable signpost interval and return its exact closer."""
        token = self._signposter.begin_interval(name)
        return lambda: self._signposter.end_interval(name, token)
