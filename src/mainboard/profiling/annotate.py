import sys
import threading
from collections.abc import Iterable
from types import CodeType
from typing import Protocol, cast

from .spans import SpanToken, finish, start
from .trace import CallbackSession
from .tracer import Tracer

_tracer: Tracer | None = None
_tool_id = sys.monitoring.PROFILER_ID
_codes: tuple[CodeType, ...] = ()
_frames = threading.local()


class ReturnedValue(Protocol):
    """An intentionally unconstrained value emitted by a monitored return."""


def tracer() -> Tracer:
    """Detect and cache the native annotation and activity backend."""
    global _tracer
    if _tracer is None:
        _tracer = Tracer.detect()
    return _tracer


def callbacks(domains: tuple[str, ...] = ("runtime", "driver")) -> CallbackSession:
    """Count selected native API callbacks through the detected backend."""
    return tracer().callbacks(domains)


def frames() -> list[SpanToken]:
    """Return the current thread's automatic span stack."""
    if not hasattr(_frames, "value"):
        _frames.value = []
    return cast(list[SpanToken], _frames.value)


def on_start(code: CodeType, offset: int) -> None:
    """Open a span for one code object selected through local monitoring."""
    del offset
    frames().append(start(code.co_qualname))


def on_return(code: CodeType, offset: int, retval: ReturnedValue) -> None:
    """Close the automatic span paired with a return event."""
    del code, offset, retval
    stack = frames()
    if stack:
        finish(stack.pop())


def on_unwind(code: CodeType, offset: int, exc: BaseException) -> None:
    """Close the automatic span paired with an exceptional unwind."""
    if code in _codes:
        on_return(code, offset, exc)


def enable_auto(codes: Iterable[CodeType]) -> None:
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
