import functools
import inspect
import time
from collections.abc import Awaitable, Callable
from types import TracebackType
from typing import Protocol, cast, overload


class SpanSession(Protocol):
    """The small surface dormant spans need from an active profiler."""

    def enter(self, name: str) -> int:
        """Open `name` and return its session-local token."""

    def exit(self, token: int, wall_ns: int) -> None:
        """Close a prior token after `wall_ns` elapsed."""


class ReturnedValue(Protocol):
    """An intentionally unconstrained decorated return value."""


_active: SpanSession | None = None
type SpanToken = tuple[SpanSession, int, int] | None


def activate(session: SpanSession) -> None:
    """Route dormant spans to `session` until it is deactivated."""
    global _active
    if _active is not None:
        raise RuntimeError("only one Profiler may be active in a process")
    _active = session


def deactivate(session: SpanSession) -> None:
    """Stop routing spans when `session` owns the active slot."""
    global _active
    if _active is session:
        _active = None


def start(name: str) -> SpanToken:
    """Start `name` when a profiler is active, otherwise return no token."""
    session = _active
    if session is None:
        return None
    token = session.enter(name)
    return session, token, time.perf_counter_ns()


def finish(token: SpanToken) -> None:
    """Close a token returned by `start`."""
    if token is None:
        return
    session, key, start_ns = token
    session.exit(key, time.perf_counter_ns() - start_ns)


class _Span:
    """A named context that is dormant until a `Profiler` is active."""

    __slots__ = ("name", "session", "start_ns", "token")

    def __init__(self, name: str) -> None:
        self.name = name
        self.session: SpanSession | None = None
        self.start_ns = 0
        self.token = 0

    def __enter__(self) -> _Span:
        token = start(self.name)
        if token is not None:
            self.session, self.token, self.start_ns = token
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None = None,
        exc: BaseException | None = None,
        traceback: TracebackType | None = None,
    ) -> None:
        session = self.session
        if session is None:
            return
        self.session = None
        finish((session, self.token, self.start_ns))

    def __call__[**P, R](self, func: Callable[P, R]) -> Callable[P, R]:
        """Decorate `func` with this span name without adding collection policy."""
        if inspect.iscoroutinefunction(func):
            return cast(
                Callable[P, R],
                _decorate_async(cast(Callable[P, Awaitable[ReturnedValue]], func), self.name),
            )
        return _decorate_sync(func, self.name)


def _decorate_sync[**P, R](func: Callable[P, R], label: str) -> Callable[P, R]:
    """Wrap a function with one active-session branch on every call."""

    @functools.wraps(func)
    def inner(*args: P.args, **kwargs: P.kwargs) -> R:
        session = _active
        if session is None:
            return func(*args, **kwargs)
        key = session.enter(label)
        start_ns = time.perf_counter_ns()
        try:
            return func(*args, **kwargs)
        finally:
            session.exit(key, time.perf_counter_ns() - start_ns)

    return inner


def _decorate_async[**P, R](
    func: Callable[P, Awaitable[R]], label: str
) -> Callable[P, Awaitable[R]]:
    """Wrap a coroutine with one active-session branch on every awaited call."""

    @functools.wraps(func)
    async def inner(*args: P.args, **kwargs: P.kwargs) -> R:
        session = _active
        if session is None:
            return await func(*args, **kwargs)
        key = session.enter(label)
        start_ns = time.perf_counter_ns()
        try:
            return await func(*args, **kwargs)
        finally:
            session.exit(key, time.perf_counter_ns() - start_ns)

    return inner


@overload
def span[**P, R](
    name: Callable[P, Awaitable[R]],
) -> Callable[P, Awaitable[R]]: ...


@overload
def span[**P, R](name: Callable[P, R]) -> Callable[P, R]: ...


@overload
def span(name: str) -> _Span: ...


def span[**P, R](name: str | Callable[P, R]) -> _Span | Callable[P, R]:
    """Mark a named block or function for an active `Profiler`.

    The annotation contains no collection policy. Without an active profiler it
    performs no clock, memory, marker, device, or context-variable work.
    """
    if isinstance(name, str):
        return _Span(name)
    label = getattr(name, "__qualname__", name.__class__.__qualname__)
    if inspect.iscoroutinefunction(name):
        return cast(
            Callable[P, R],
            _decorate_async(cast(Callable[P, Awaitable[ReturnedValue]], name), label),
        )
    return _decorate_sync(name, label)
