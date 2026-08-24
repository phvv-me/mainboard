import sys
import threading
import types

import pytest

from mainboard import Profiler
from mainboard.profile import Tracer, annotate


def test_the_backend_is_detected_once_with_the_vendors_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`present` reaches `Tracer.detect` unchanged and the instance is cached from then on.

    Only the first caller's `present` decides the backend, since detection never repeats
    for the life of the process, and `callbacks()` reads through the same cached instance.
    """
    seen: list[frozenset[str]] = []

    def detect(cls: type[Tracer], *, present: frozenset[str] = frozenset()) -> Tracer:
        seen.append(present)
        return Tracer()

    monkeypatch.setattr(Tracer, "detect", classmethod(detect))
    first = annotate.tracer(present=frozenset({"nvidia"}))
    assert annotate.tracer() is first
    assert seen == [frozenset({"nvidia"})]
    with annotate.callbacks() as session:
        assert session.counts() == {}


def test_enable_auto_instruments_matching_calls() -> None:
    """Runtime auto-annotation installs local events for selected code only."""
    calls: list[str] = []

    def target() -> int:
        calls.append("start")
        return 41

    annotate.frames().clear()
    annotate.enable_auto((target.__code__,))
    try:
        assert target() == 41
    finally:
        annotate.disable_auto()
    assert calls == ["start"]
    assert annotate.enabled_codes() == ()


def test_module_codes_finds_owned_code_and_nothing_else(monkeypatch: pytest.MonkeyPatch) -> None:
    """Auto-annotation covers a module's own functions, never the ones it imported."""
    empty = types.ModuleType("mainboard_fake_module")
    monkeypatch.setitem(sys.modules, empty.__name__, empty)
    assert Profiler.module_codes((empty.__name__,)) == set()

    module = types.ModuleType("mainboard_owned_module")

    def owned() -> None:
        pass

    def imported() -> None:
        pass

    owned.__module__ = module.__name__
    vars(module).update(owned=owned, imported=imported)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    assert Profiler.module_codes((module.__name__,)) == {owned.__code__}


def test_monitor_hooks_balance_return_unwind_and_empty_stack(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every close pairs with an open and never pops past the bottom of the stack.

    An unwind of code that was never selected closes nothing, and a return against an
    already empty stack is ignored.
    """
    selected = (lambda: None).__code__
    other = (lambda: 1).__code__
    monkeypatch.setattr(annotate, "_codes", (selected,))
    annotate.frames().clear()

    annotate.on_start(selected, 0)
    annotate.on_start(selected, 0)
    annotate.on_unwind(other, 0, ValueError())  # not selected, so it closes nothing
    assert len(annotate.frames()) == 2
    annotate.on_return(selected, 0, None)
    annotate.on_unwind(selected, 0, ValueError())
    annotate.on_return(selected, 0, None)  # already empty
    assert annotate.frames() == []


def test_monitor_stack_is_thread_local() -> None:
    """A thread that never opened a span starts from its own empty stack."""
    seen: list[int] = []
    thread = threading.Thread(target=lambda: seen.append(len(annotate.frames())))
    thread.start()
    thread.join(timeout=5)
    assert seen == [0]
