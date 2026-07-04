"""Annotation surface: manual `region`/`@profile` and AST/`sys.monitoring` auto.

One API works whether or not a :class:`~mainboard.profiling.profiler.Profiler` is
running: a region always emits the active :class:`Tracer`'s native annotation, and
when a profiler is active it also brackets timing + snapshots. Auto-annotation comes
two ways: ``enable_auto`` instruments every Python call at runtime via PEP 669
(``sys.monitoring``, zero source edits), and ``instrument_source`` statically rewrites
a module's source to wrap each function body in a ``region`` (the inspectable path).
"""

import ast
import functools
import sys
import threading
import time
from collections.abc import Callable
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, overload

from .tracer import Tracer

if TYPE_CHECKING:
    from collections.abc import Generator
    from types import CodeType

    from .profiler import Profiler
    from .trace import CallbackSession

type CloseToken = tuple["Profiler", str, int] | None

_tracer: Tracer | None = None
profiler: Profiler | None = None  # set by Profiler.__enter__; read here to bracket timing


def tracer() -> Tracer:
    """The process-wide active tracer, lazily detected on first use."""
    global _tracer
    if _tracer is None:
        _tracer = Tracer.detect()
    return _tracer


def callbacks(domains: tuple[str, ...] = ("runtime", "driver")) -> CallbackSession:
    """Count CUDA API calls by name, synchronously: ``with callbacks() as cb: ...``.

    Then read ``cb.counts()``. Wraps the active tracer's CUPTI Callback API.
    """
    return tracer().callbacks(domains)


def _enter(name: str) -> CloseToken:
    """Open a region: native annotation always, timing only when a profiler is active.

    Returns a close token, or ``None`` when no profiler is running — so the disabled
    path skips the clock read and timing entirely (minimal impact).
    """
    tracer().push(name)
    active = profiler
    if active is None:
        return None
    active.enter(name)
    return active, name, time.perf_counter_ns()


def _exit(token: CloseToken) -> None:
    """Close a region opened by :func:`_enter`."""
    if token is not None:
        active, name, start = token
        active.exit(name, time.perf_counter_ns() - start)
    tracer().pop()


@contextmanager
def region(name: str) -> Generator[None]:
    """Annotate a named region (and time it when a profiler is active)."""
    token = _enter(name)
    try:
        yield
    finally:
        _exit(token)


@overload
def profile[**P, R](fn: Callable[P, R]) -> Callable[P, R]: ...


@overload
def profile[**P, R](
    fn: None = None, *, name: str | None = None
) -> Callable[[Callable[P, R]], Callable[P, R]]: ...


def profile[**P, R](
    fn: Callable[P, R] | None = None, *, name: str | None = None
) -> Callable[P, R] | Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorate a function so each call is a profiled region (default name: qualname).

    Use bare (`@profile`) or with a name (`@profile(name="...")`).
    """

    def wrap(func: Callable[P, R]) -> Callable[P, R]:
        label = name or func.__qualname__

        @functools.wraps(func)
        def inner(*args: P.args, **kwargs: P.kwargs) -> R:
            with region(label):
                return func(*args, **kwargs)

        return inner

    return wrap(fn) if fn is not None else wrap


# ── runtime auto-annotation via PEP 669 (sys.monitoring) ────────────────────────

_TOOL_ID = sys.monitoring.PROFILER_ID
_predicate: Callable[[CodeType], bool] | None = None
_frames = threading.local()


def _stack() -> list[CloseToken]:
    if not hasattr(_frames, "value"):
        _frames.value = []
    stack: list[CloseToken] = _frames.value
    return stack


def _on_start(code: CodeType, _offset: int) -> None:
    matched = _predicate is not None and _predicate(code)
    _stack().append(_enter(code.co_qualname) if matched else None)


# `sys.monitoring` PY_RETURN hands back the function's return value, of any type, which
# we never inspect; CPython exposes no narrower type for it (irreducible `Any`).
def _on_return(code: CodeType, _offset: int, _retval: Any) -> None:
    stack = _stack()
    if stack:
        token = stack.pop()
        if token is not None:
            _exit(token)


def _on_unwind(code: CodeType, _offset: int, _exc: BaseException) -> None:
    _on_return(code, _offset, None)


def enable_auto(predicate: Callable[[CodeType], bool]) -> None:
    """Auto-annotate every Python call whose code object satisfies ``predicate``.

    predicate: e.g. ``lambda c: c.co_filename.startswith(root)`` to scope to a package.
    Zero source edits; balanced across returns and exceptions. Generators that suspend
    across yields are not perfectly bracketed (a known PEP 669 limitation).
    """
    global _predicate
    _predicate = predicate
    monitor = sys.monitoring
    monitor.use_tool_id(_TOOL_ID, "mainboard")
    events = monitor.events
    monitor.register_callback(_TOOL_ID, events.PY_START, _on_start)
    monitor.register_callback(_TOOL_ID, events.PY_RETURN, _on_return)
    monitor.register_callback(_TOOL_ID, events.PY_UNWIND, _on_unwind)
    monitor.set_events(_TOOL_ID, events.PY_START | events.PY_RETURN | events.PY_UNWIND)


def disable_auto() -> None:
    """Stop runtime auto-annotation and release the monitoring tool id."""
    global _predicate
    monitor = sys.monitoring
    monitor.set_events(_TOOL_ID, 0)
    monitor.free_tool_id(_TOOL_ID)
    _predicate = None


# ── static auto-annotation via AST rewrite ──────────────────────────────────────

_IMPORT = "from mainboard.profiling import region as _mb_region"


class _Injector(ast.NodeTransformer):
    """Wrap each function body in `with region("<qualname>"): ...`."""

    def __init__(self) -> None:
        self._scope: list[str] = []

    def _wrap(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> ast.AST:
        self._scope.append(node.name)
        self.generic_visit(node)
        qualname = ".".join(self._scope)
        self._scope.pop()
        call = ast.Call(
            func=ast.Name(id="_mb_region", ctx=ast.Load()),
            args=[ast.Constant(value=qualname)],
            keywords=[],
        )
        item = ast.withitem(context_expr=call, optional_vars=None)
        node.body = [ast.With(items=[item], body=node.body)]
        return node

    visit_FunctionDef = _wrap
    visit_AsyncFunctionDef = _wrap


def instrument_source(source: str) -> str:
    """Rewrite module ``source`` so every function body runs inside a `region`.

    Returns new source (with the `region` import prepended) for inspection or exec;
    the static counterpart to :func:`enable_auto`.
    """
    tree = _Injector().visit(ast.parse(source))
    ast.fix_missing_locations(tree)
    return f"{_IMPORT}\n{ast.unparse(tree)}"
