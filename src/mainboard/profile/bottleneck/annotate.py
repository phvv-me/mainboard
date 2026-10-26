import sys
from contextvars import ContextVar
from typing import TYPE_CHECKING

from ..spans import finish, start
from ..trace import CallbackSession
from ..tracer import Tracer

if TYPE_CHECKING:
    from collections.abc import Iterable
    from types import CodeType

    from ..spans import SpanToken

_tracer: Tracer | None = None
_tool_id = sys.monitoring.PROFILER_ID
_codes: tuple[CodeType, ...] = ()
_frames: ContextVar[list[SpanToken] | None] = ContextVar("mainboard_auto_spans", default=None)


def tracer(*, present: frozenset[str] = frozenset()) -> Tracer:
    """Detect and cache the native annotation and activity backend.

    present: vendors on this host (`GPU.vendor` values). Only the first call's value
        counts, since detection never repeats for the life of the process.
    """
    global _tracer
    if _tracer is None:
        _tracer = Tracer.detect(present=present)
    return _tracer


def callbacks(domains: tuple[str, ...] = ("runtime", "driver")) -> CallbackSession:
    """Count selected native API callbacks through the detected backend."""
    return tracer().callbacks(domains)


def _open_frames() -> list[SpanToken]:
    """Return the current task's automatic span stack."""
    stack = _frames.get()
    if stack is None:
        stack = []
        _frames.set(stack)
    return stack


def enabled_codes() -> tuple[CodeType, ...]:
    """Return the code objects currently selected for local PEP 669 monitoring."""
    return _codes


def _on_start(code: CodeType, offset: int) -> None:
    """Open a span for one code object selected through local monitoring."""
    del offset
    _frames.set([*_open_frames(), start(code.co_qualname)])


def _close() -> None:
    """Close the automatic span a return, yield or exceptional unwind ends."""
    stack = _open_frames()
    if stack:
        finish(stack[-1])
        _frames.set(stack[:-1])


def _on_unwind(code: CodeType, offset: int, exc: BaseException) -> None:
    """Close the automatic span paired with an exceptional unwind."""
    del offset, exc
    if code in _codes:
        _close()


def _register_callbacks() -> None:
    """Route the profiler tool's PEP 669 events to this module's span callbacks."""
    monitor = sys.monitoring
    events = monitor.events
    monitor.register_callback(_tool_id, events.PY_START, _on_start)
    monitor.register_callback(_tool_id, events.PY_RESUME, _on_start)
    # A return or yield event passes the code, offset and value, none of which the span needs.
    monitor.register_callback(_tool_id, events.PY_RETURN, lambda *_: _close())
    monitor.register_callback(_tool_id, events.PY_YIELD, lambda *_: _close())
    monitor.register_callback(_tool_id, events.PY_UNWIND, _on_unwind)


def enable_auto(codes: Iterable[CodeType]) -> None:
    """Enable PEP 669 events only on `codes`, sparing every other function a callback."""
    global _codes
    _codes = tuple(dict.fromkeys(codes))
    monitor = sys.monitoring
    monitor.use_tool_id(_tool_id, "mainboard")
    _register_callbacks()
    events = monitor.events
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
