import types
from typing import Any

import pytest

from mainboard.profile.providers import apple_tracer


def test_signpost_tracer_intervals(monkeypatch: pytest.MonkeyPatch) -> None:
    """The signpost backend opens/closes intervals via a (name, token) stack."""
    calls: list[tuple[str, Any]] = []

    class FakeSignposter:
        def __init__(self, subsystem: str) -> None:
            calls.append(("init", subsystem))

        def begin_interval(self, name: str) -> str:
            calls.append(("begin", name))
            return f"tok:{name}"

        def emit_event(self, name: str) -> None:
            calls.append(("event", name))

        def end_interval(self, name: str, token: str) -> None:
            calls.append(("end", token))

    monkeypatch.setattr(
        apple_tracer, "_signpost", types.SimpleNamespace(Signposter=FakeSignposter)
    )
    monkeypatch.setattr(apple_tracer.platform, "system", lambda: "Darwin")
    tracer = apple_tracer.SignpostTracer()
    assert apple_tracer.SignpostTracer.is_available() is True
    tracer.push("a")
    tracer.mark("e")
    tracer.pop()
    tracer.pop()  # empty stack: ignored
    finish_first = tracer.start("first")
    finish_second = tracer.start("second")
    finish_first()
    finish_second()
    assert ("begin", "a") in calls and ("event", "e") in calls


def test_signpost_unavailable_without_the_library(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(apple_tracer, "_signpost", None)
    assert apple_tracer.SignpostTracer.is_available() is False


def test_signpost_unavailable_off_darwin(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(apple_tracer, "_signpost", types.SimpleNamespace(Signposter=object))
    monkeypatch.setattr(apple_tracer.platform, "system", lambda: "Linux")
    assert apple_tracer.SignpostTracer.is_available() is False
