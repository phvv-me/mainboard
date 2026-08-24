import types

import pytest

from mainboard.profile.providers import apple_tracer


def test_the_signpost_backend_pairs_intervals_and_needs_darwin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Push and pop keep a (name, token) stack, and `start` closes the interval it opened.

    Popping an empty stack is ignored, and the backend is unavailable both without the
    `os_signpost` package and on any platform other than Darwin, since Instruments is the
    only thing that reads these signposts.
    """
    calls: list[tuple[str, str]] = []

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
    assert calls == [
        ("init", "me.phvv.mainboard"),
        ("begin", "a"),
        ("event", "e"),
        ("end", "tok:a"),
        ("begin", "first"),
        ("begin", "second"),
        ("end", "tok:first"),  # each closer ends its own interval, not the newest one
        ("end", "tok:second"),
    ]

    monkeypatch.setattr(apple_tracer.platform, "system", lambda: "Linux")
    assert apple_tracer.SignpostTracer.is_available() is False
    monkeypatch.setattr(apple_tracer, "_signpost", None)
    assert apple_tracer.SignpostTracer.is_available() is False
