import functools
import inspect
import time
from typing import TYPE_CHECKING, Protocol, cast, overload

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable, Coroutine, Iterator
    from types import TracebackType


class SpanSession(Protocol):
    """The small surface dormant spans need from an active profiler."""

    def enter(self, name: str) -> int:
        """Open `name` and return its session-local token."""

    def exit(self, token: int, *, wall_ns: int) -> None:
        """Close a prior token after `wall_ns` elapsed."""


class ReturnedValue(Protocol):
    """An intentionally unconstrained decorated return value."""


_active: SpanSession | None = None
type SpanToken = tuple[SpanSession, int, int] | None


def active() -> SpanSession | None:
    """Return the profiler currently receiving spans, or None if none is active."""
    return _active


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
    session.exit(key, wall_ns=time.perf_counter_ns() - start_ns)


class _Span:
    """A named context that is dormant until a `Profiler` is active."""

    __slots__ = ("name", "session", "start_ns", "token")

    def __init__(self, name: str) -> None:
        self.name = name
        self.session: SpanSession | None = None
        self.start_ns = 0
        self.token = 0

    def __call__[**P, R](self, func: Callable[P, R]) -> Callable[P, R]:
        """Decorate `func` with this span name without adding collection policy.

        Generator and async generator functions get consumption-spanning
        wrappers, since timing only the generator object's creation is the
        silent zero-length-span bug CPython fixed for its own decorators.
        """
        if inspect.iscoroutinefunction(func):
            return cast(
                "Callable[P, R]",
                _decorate_async(cast("Callable[P, Awaitable[ReturnedValue]]", func), self.name),
            )
        if inspect.isasyncgenfunction(func):
            return cast("Callable[P, R]", _decorate_async_generator(func, self.name))
        if inspect.isgeneratorfunction(func):
            return cast("Callable[P, R]", _decorate_generator(func, self.name))
        return _decorate_sync(func, self.name)

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
            session.exit(key, wall_ns=time.perf_counter_ns() - start_ns)

    return inner


def _decorate_async[**P, R](
    func: Callable[P, Awaitable[R]], label: str
) -> Callable[P, Coroutine[object, object, R]]:
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
            session.exit(key, wall_ns=time.perf_counter_ns() - start_ns)

    return inner


def _decorate_generator[**P, Y](
    func: Callable[P, Iterator[Y]], label: str
) -> Callable[P, Iterator[Y]]:
    """Wrap a generator function so the span covers consumption, not creation.

    The session branch runs at first iteration, so a dormant span still costs
    one module-global read, and `yield from` keeps send, throw, and close
    delegation intact.
    """

    @functools.wraps(func)
    def inner(*args: P.args, **kwargs: P.kwargs) -> Iterator[Y]:
        session = _active
        if session is None:
            yield from func(*args, **kwargs)
            return
        key = session.enter(label)
        start_ns = time.perf_counter_ns()
        try:
            yield from func(*args, **kwargs)
        finally:
            session.exit(key, wall_ns=time.perf_counter_ns() - start_ns)

    return inner


def _decorate_async_generator[**P, Y](
    func: Callable[P, AsyncIterator[Y]], label: str
) -> Callable[P, AsyncIterator[Y]]:
    """Wrap an async generator so the span covers consumption, not creation.

    Delegation is a plain async-for, so value-sending and throw forwarding
    through the async protocol are not preserved, the same fidelity
    CPython's own decorator fix settled on.
    """

    @functools.wraps(func)
    async def inner(*args: P.args, **kwargs: P.kwargs) -> AsyncIterator[Y]:
        session = _active
        if session is None:
            async for item in func(*args, **kwargs):
                yield item
            return
        key = session.enter(label)
        start_ns = time.perf_counter_ns()
        try:
            async for item in func(*args, **kwargs):
                yield item
        finally:
            session.exit(key, wall_ns=time.perf_counter_ns() - start_ns)

    return inner


@overload
def span[**P, R](
    name: Callable[P, Coroutine[object, object, R]],
) -> Callable[P, Coroutine[object, object, R]]: ...


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
    return _Span(getattr(name, "__qualname__", name.__class__.__qualname__))(name)
