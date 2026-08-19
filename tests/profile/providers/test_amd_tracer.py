import types
from typing import Any

import pytest

from mainboard.profile.providers import amd_tracer


def test_roctx_tracer_push_pop_mark(monkeypatch: pytest.MonkeyPatch) -> None:
    """The ROCTx backend tracks a range-id stack and forwards marks."""
    events: list[tuple[str, Any]] = []
    fake = types.SimpleNamespace(
        rangeStart=lambda name: events.append(("start", name)) or len(events),
        rangeStop=lambda rid: events.append(("stop", rid)),
        mark=lambda name: events.append(("mark", name)),
    )
    monkeypatch.setattr(amd_tracer, "roctx", fake)
    tracer = amd_tracer.RoctxTracer()
    assert amd_tracer.RoctxTracer.is_available() is True
    tracer.push("r")
    tracer.mark("m")
    tracer.pop()
    tracer.pop()  # empty stack: ignored
    finish_first = tracer.start("first")
    finish_second = tracer.start("second")
    finish_first()
    finish_second()
    assert ("start", "r") in events and ("mark", "m") in events
    monkeypatch.setattr(amd_tracer, "roctx", None)
    tracer.start("unavailable")()


def test_roctx_unavailable_when_library_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(amd_tracer, "roctx", None)
    assert amd_tracer.RoctxTracer.is_available() is False
