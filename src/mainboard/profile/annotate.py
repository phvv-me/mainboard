import sys
from contextvars import ContextVar
from typing import TYPE_CHECKING, Protocol

from .spans import finish, start
from .trace import CallbackSession
from .tracer import Tracer

if TYPE_CHECKING:
    from collections.abc import Iterable
    from types import CodeType

    from .spans import SpanToken

_tracer: Tracer | None = None
_tool_id = sys.monitoring.PROFILER_ID
_codes: tuple["CodeType", ...] = ()
_frames: ContextVar[list["SpanToken"] | None] = ContextVar("mainboard_auto_spans", default=None)


class ReturnedValue(Protocol):
    """An intentionally unconstrained value emitted by a monitored return."""


def tracer(*, present: frozenset[str] = frozenset()) -> Tracer:
    """Detect and cache the native annotation and activity backend.

    present: hardware vendors actually on this host (`DeviceProbe.vendor` values), read
        once by the first caller; only that first call's `present` decides the cached
        backend, since detection then never repeats for the life of the process.
    """
    global _tracer
    if _tracer is None:
        _tracer = Tracer.detect(present=present)
    return _tracer


def callbacks(domains: tuple[str, ...] = ("runtime", "driver")) -> CallbackSession:
    """Count selected native API callbacks through the detected backend."""
    return tracer().callbacks(domains)


def frames() -> list["SpanToken"]:
    """Return the current task's automatic span stack."""
    stack = _frames.get()
    if stack is None:
        stack = []
        _frames.set(stack)
    return stack


def enabled_codes() -> tuple["CodeType", ...]:
    """Return the code objects currently selected for local PEP 669 monitoring."""
    return _codes


def on_start(code: "CodeType", offset: int) -> None:
    """Open a span for one code object selected through local monitoring."""
    del offset
    _frames.set([*frames(), start(code.co_qualname)])


def on_return(code: "CodeType", offset: int, retval: ReturnedValue) -> None:
    """Close the automatic span paired with a return event."""
    del code, offset, retval
    stack = frames()
    if stack:
        finish(stack[-1])
        _frames.set(stack[:-1])


def on_unwind(code: "CodeType", offset: int, exc: BaseException) -> None:
    """Close the automatic span paired with an exceptional unwind."""
    if code in _codes:
        on_return(code, offset, exc)


def enable_auto(codes: "Iterable[CodeType]") -> None:
    """Enable PEP 669 events only on explicit code objects.

    Local events avoid a Python predicate callback on every function in the process.
    """
    global _codes
    _codes = tuple(dict.fromkeys(codes))
    monitor = sys.monitoring
    monitor.use_tool_id(_tool_id, "mainboard")
    events = monitor.events
    monitor.register_callback(_tool_id, events.PY_START, on_start)
    monitor.register_callback(_tool_id, events.PY_RESUME, on_start)
    monitor.register_callback(_tool_id, events.PY_RETURN, on_return)
    monitor.register_callback(_tool_id, events.PY_YIELD, on_return)
    monitor.register_callback(_tool_id, events.PY_UNWIND, on_unwind)
    monitor.set_events(_tool_id, events.PY_UNWIND)
    selected = events.PY_START | events.PY_RESUME | events.PY_RETURN | events.PY_YIELD
    for code in _codes:
        monitor.set_local_events(_tool_id, code, selected)


def disable_auto() -> None:
    """Remove every local monitoring event and release the tool identifier."""
    global _codes
    monitor = sys.monitoring
    monitor.set_events(_tool_id, 0)
    for code in _codes:
        monitor.set_local_events(_tool_id, code, 0)
    monitor.free_tool_id(_tool_id)
    _codes = ()
