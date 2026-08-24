import types

import pytest

from mainboard.profile.providers import amd_tracer


def test_the_roctx_backend_tracks_a_range_id_stack_and_needs_the_library(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ROCTx push/pop keep a range-id stack, and `start` hands back that range's own closer.

    Popping an empty stack is ignored rather than an error, and with ROCTx missing the
    backend reports itself unavailable and its markers degrade to no-ops.
    """
    events: list[tuple[str, str | int]] = []
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
    assert events == [
        ("start", "r"),
        ("mark", "m"),
        ("stop", 1),
        ("start", "first"),
        ("start", "second"),
        ("stop", 4),  # each closer stops its own range, not the newest one
        ("stop", 5),
    ]

    monkeypatch.setattr(amd_tracer, "roctx", None)
    assert amd_tracer.RoctxTracer.is_available() is False
    tracer.start("unavailable")()
