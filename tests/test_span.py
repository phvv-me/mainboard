import threading
import time

import pytest

from mainboard.profiling import Profile, Profiler, span, spans


def test_dormant_context_and_decorators_do_no_collection() -> None:
    """Annotations call through normally when no profiler owns the process."""

    @span
    def bare(value: int) -> int:
        return value + 1

    @span("named")
    def named(value: int) -> int:
        return value * 2

    with span("context"):
        assert bare(1) == 2
        assert named(2) == 4
    assert spans._active is None


def test_dormant_annotations_do_not_read_a_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    """The inactive path avoids clocks and every collector side effect."""

    def fail() -> int:
        raise AssertionError("clock read")

    monkeypatch.setattr(spans.time, "perf_counter_ns", fail)

    @span
    def work() -> int:
        return 7

    with span("context"):
        assert work() == 7


def test_one_profiler_collects_nested_context_and_decorator_spans() -> None:
    """All annotation forms feed one profile with dotted nesting paths."""

    @span("child")
    def work() -> None:
        time.sleep(0.0001)

    with Profiler(features=Profiler.Feature.SPANS) as profiler, span("parent"):
        work()

    profile = profiler.result()
    assert isinstance(profile, Profile)
    assert [item.name for item in profile.summaries] == ["parent.child", "parent"]
    assert all(item.wall_ms > 0 for item in profile.summaries)


def test_exception_still_closes_the_span() -> None:
    profiler = Profiler(features=Profiler.Feature.SPANS)
    with profiler, pytest.raises(ValueError, match="boom"), span("risky"):
        raise ValueError("boom")
    assert profiler.result().summaries[0].name == "risky"


def test_only_one_profiler_can_receive_process_spans() -> None:
    first = Profiler(features=Profiler.Feature.SPANS)
    second = Profiler(features=Profiler.Feature.SPANS)
    with first, pytest.raises(RuntimeError, match="only one"):
        second.__enter__()
    assert second.active is False


def test_profiler_instance_cannot_be_entered_twice() -> None:
    profiler = Profiler(features=Profiler.Feature.SPANS)
    with profiler, pytest.raises(RuntimeError, match="entered twice"):
        profiler.__enter__()


def test_span_buffer_is_bounded_and_reports_drops() -> None:
    with Profiler(features=Profiler.Feature.SPANS, max_spans=2) as profiler:
        for name in ("one", "two", "three"):
            with span(name):
                pass
    profile = profiler.result()
    assert [item.name for item in profile.summaries] == ["two", "three"]
    assert profile.dropped_spans == 1
    assert "oldest spans dropped" in profile.report()


def test_concurrent_threads_close_their_exact_tokens() -> None:
    ready = threading.Barrier(8)

    @span("worker")
    def work() -> None:
        ready.wait(timeout=5)

    with Profiler(features=Profiler.Feature.SPANS) as profiler:
        threads = [threading.Thread(target=work) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)

    assert len(profiler.result().summaries) == 8
    assert {item.name for item in profiler.result().summaries} == {"worker"}


def test_deactivate_ignores_a_session_that_does_not_own_the_slot() -> None:
    first = Profiler(features=Profiler.Feature.SPANS)
    second = Profiler(features=Profiler.Feature.SPANS)
    spans.activate(first)
    spans.deactivate(second)
    assert spans._active is first
    spans.deactivate(first)


def test_finish_and_inactive_exit_are_safe_noops() -> None:
    spans.finish(None)
    context = span("unused")
    context.__exit__(None, None, None)
