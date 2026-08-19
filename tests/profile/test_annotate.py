import sys
import threading
import types

import pytest

from mainboard import Profiler
from mainboard.profile import Tracer, annotate

from .conftest import clock_tracer


def test_tracer_lazy_detection_is_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    """`annotate.tracer()` detects once and caches the instance."""
    monkeypatch.setattr(Tracer, "detect", classmethod(lambda cls, **_: Tracer()))
    first = annotate.tracer()
    assert annotate.tracer() is first


def test_tracer_forwards_present_vendors_to_detect(monkeypatch: pytest.MonkeyPatch) -> None:
    """`present` reaches `Tracer.detect` unchanged, so the first caller decides the backend."""
    seen = []
    monkeypatch.setattr(
        Tracer, "detect", classmethod(lambda cls, *, present=frozenset(): seen.append(present))
    )
    annotate.tracer(present=frozenset({"nvidia"}))
    assert seen == [frozenset({"nvidia"})]


def test_callbacks_proxies_the_tracer(monkeypatch: pytest.MonkeyPatch) -> None:
    """`callbacks()` returns the active tracer's callback session."""
    monkeypatch.setattr(annotate, "_tracer", clock_tracer())
    with annotate.callbacks() as session:
        pass
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


def test_module_codes_handles_missing_file(monkeypatch: pytest.MonkeyPatch) -> None:
    module = types.ModuleType("mainboard_fake_module")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    assert Profiler.module_codes((module.__name__,)) == set()


def test_module_codes_excludes_imported_functions(monkeypatch: pytest.MonkeyPatch) -> None:
    module = types.ModuleType("mainboard_owned_module")

    def owned() -> None:
        pass

    def imported() -> None:
        pass

    owned.__module__ = module.__name__
    vars(module).update(owned=owned, imported=imported)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    assert Profiler.module_codes((module.__name__,)) == {owned.__code__}


def test_monitor_hooks_balance_return_unwind_and_empty_stack() -> None:
    code = (lambda: None).__code__
    annotate.frames().clear()
    annotate.on_start(code, 0)
    annotate.on_start(code, 0)
    annotate.on_return(code, 0, None)
    annotate.on_unwind(code, 0, ValueError())
    annotate.on_return(code, 0, None)
    annotate.on_return(code, 0, None)
    assert annotate.frames() == []


def test_monitor_unwind_closes_only_selected_code(monkeypatch: pytest.MonkeyPatch) -> None:
    selected = (lambda: None).__code__
    other = (lambda: 1).__code__
    monkeypatch.setattr(annotate, "_codes", (selected,))
    annotate.frames().clear()
    annotate.on_start(selected, 0)
    annotate.on_unwind(other, 0, ValueError())
    assert len(annotate.frames()) == 1
    annotate.on_unwind(selected, 0, ValueError())
    assert annotate.frames() == []


def test_monitor_stack_is_thread_local() -> None:
    seen: list[int] = []
    thread = threading.Thread(target=lambda: seen.append(len(annotate.frames())))
    thread.start()
    thread.join(timeout=5)
    assert seen == [0]
