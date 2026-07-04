"""`span`: a lightweight, nestable timing/memory measurement, off by default.

Unlike `Profiler` (a bounded session that samples GPU telemetry on a background
thread), a span is meant to stay wired into production code: `with span("extract"):`
or `@span` costs one boolean check when disabled, and only times the block — and, on
request, its process/GPU memory delta — once `enable_spans()` has been called. Nesting
is tracked with a `ContextVar` stack, so each asyncio task, and each thread, gets its
own uncontaminated stack: the same code works run sequentially or fanned out across
many concurrent tasks. Every closed span folds into a `Collector`: the process-wide
`default_collector()` by default, or one the caller owns for a scoped window (one
`Collector` per recall call, say).
"""

import functools
import inspect
import time
from collections.abc import Callable, Coroutine
from contextvars import ContextVar
from functools import cache
from types import TracebackType
from typing import Any, cast, overload

import psutil

from ..gpu import GPU
from .collector import Collector, default_collector
from .models import SpanRecord

_stack: ContextVar[tuple[str, ...]] = ContextVar("mainboard_span_stack", default=())
_process = psutil.Process()
_span_enabled = False


def enable_spans() -> None:
    """Turn on span timing/memory recording process-wide (default: off)."""
    global _span_enabled
    _span_enabled = True


def disable_spans() -> None:
    """Turn off span recording; `span(...)` then costs one truthiness check."""
    global _span_enabled
    _span_enabled = False


def spans_enabled() -> bool:
    """Whether span recording is currently on."""
    return _span_enabled


@cache
def _gpu_units() -> tuple[GPU, ...]:
    """The machine's GPUs, probed once and cached for span memory deltas."""
    return GPU.all()


def _gpu_used_bytes() -> int:
    """Summed used memory across every detected GPU, or 0 on a CPU-only host."""
    return sum(gpu.memory.used_bytes for gpu in _gpu_units())


def _decorate_sync[**P, R](
    func: Callable[P, R], label: str, collector: Collector | None, memory: bool
) -> Callable[P, R]:
    """Wrap a sync function so each call runs inside a fresh `span` of `label`."""

    @functools.wraps(func)
    def inner(*args: P.args, **kwargs: P.kwargs) -> R:
        with span(label, collector=collector, memory=memory):
            return func(*args, **kwargs)

    return inner


def _decorate_async[**P, R](
    func: Callable[P, Coroutine[Any, Any, R]],
    label: str,
    collector: Collector | None,
    memory: bool,
) -> Callable[P, Coroutine[Any, Any, R]]:
    """Wrap a coroutine function so each call runs inside a fresh `span` of `label`."""

    @functools.wraps(func)
    async def inner(*args: P.args, **kwargs: P.kwargs) -> R:
        with span(label, collector=collector, memory=memory):
            return await func(*args, **kwargs)

    return inner


class Span:
    """A named, nestable timing (and optional memory) measurement.

    Use as a context manager (`with span("extract"):`) or as a decorator
    (`@span("extract")`, or bare `@span` to use the function's qualname). Recording is
    a no-op unless `enable_spans()` was called: `__enter__`/`__exit__` do nothing but
    read the module switch, so a disabled span reads no `ContextVar`, takes no
    timestamp, and allocates no record.
    """

    __slots__ = (
        "name",
        "collector",
        "memory",
        "_active",
        "_start_ns",
        "_path",
        "_depth",
        "_token",
        "_rss_before",
        "_gpu_before",
    )

    def __init__(
        self, name: str, *, collector: Collector | None = None, memory: bool = False
    ) -> None:
        self.name = name
        self.collector = collector
        self.memory = memory
        self._active = False

    def __enter__(self) -> Span:
        if not _span_enabled:
            return self
        parents = _stack.get()
        self._depth = len(parents)
        self._path = ".".join((*parents, self.name))
        self._token = _stack.set((*parents, self.name))
        self._rss_before = _process.memory_info().rss if self.memory else 0
        self._gpu_before = _gpu_used_bytes() if self.memory else 0
        self._start_ns = time.perf_counter_ns()
        self._active = True
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None = None,
        exc: BaseException | None = None,
        traceback: TracebackType | None = None,
    ) -> None:
        if not self._active:
            return
        self._active = False
        wall_ns = time.perf_counter_ns() - self._start_ns
        _stack.reset(self._token)
        record = SpanRecord(
            name=self.name,
            path=self._path,
            depth=self._depth,
            wall_ms=wall_ns / 1e6,
            rss_delta_bytes=_process.memory_info().rss - self._rss_before if self.memory else None,
            gpu_delta_bytes=_gpu_used_bytes() - self._gpu_before if self.memory else None,
        )
        (self.collector or default_collector()).add(record)

    # `func` covers both a plain callable and a coroutine function; the two module-level
    # `overload`s on `span` are the typed public contract, so the dispatch here only needs
    # to route by `inspect.iscoroutinefunction` and hand off to the matching, precisely
    # typed helper (irreducible `Any`: the coroutine's send/yield types, never inspected).
    def __call__(self, func: Callable[..., Any]) -> Callable[..., Any]:
        """Decorate `func` so each call runs inside a fresh span of this name.

        A new `Span` opens per call (never this instance) so concurrent calls of the
        same decorated function never share timing/stack state.
        """
        label, collector, memory = self.name, self.collector, self.memory
        if inspect.iscoroutinefunction(func):
            return _decorate_async(
                cast(Callable[..., Coroutine[Any, Any, Any]], func), label, collector, memory
            )
        return _decorate_sync(func, label, collector, memory)


@overload
def span[**P, R](
    name: Callable[P, Coroutine[Any, Any, R]],
) -> Callable[P, Coroutine[Any, Any, R]]: ...
@overload
def span[**P, R](name: Callable[P, R]) -> Callable[P, R]: ...
@overload
def span(name: str, *, collector: Collector | None = None, memory: bool = False) -> Span: ...
def span(
    name: str | Callable[..., Any],
    *,
    collector: Collector | None = None,
    memory: bool = False,
) -> Span | Callable[..., Any]:
    """Open a span, or decorate a function so each call is one.

    `with span("extract"):` times/annotates the block. Bare `@span` decorates a
    function using its qualname; `@span("extract", memory=True)` decorates with an
    explicit label and also records the process/GPU memory delta. `collector=` sends
    the recorded spans to a caller-owned `Collector` instead of the process-wide
    default.
    """
    if isinstance(name, str):
        return Span(name, collector=collector, memory=memory)
    func = name
    label = func.__qualname__
    if inspect.iscoroutinefunction(func):
        return _decorate_async(
            cast(Callable[..., Coroutine[Any, Any, Any]], func), label, collector, memory
        )
    return _decorate_sync(func, label, collector, memory)
