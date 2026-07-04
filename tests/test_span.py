"""The `span` timing/memory API: enable switch, nesting, decorators, and `Collector`
aggregation. No GPU needed — memory tracking is exercised against the fake GPU
fixtures from `conftest`, and everything else is plain wall-clock/contextvar plumbing.
"""

import math
import threading
import time
from random import Random

import pytest
from hypothesis import given
from hypothesis import strategies as st

from mainboard.profiling import spans as span_module
from mainboard.profiling.collector import Collector, Reservoir, default_collector
from mainboard.profiling.models import SpanRecord
from mainboard.profiling.spans import Span, disable_spans, enable_spans, span, spans_enabled

names = st.sampled_from(("gate", "extract", "consolidate", "embed", "write", "a", "b"))
wall_times = st.floats(min_value=0.0, max_value=1_000.0, allow_nan=False, allow_infinity=False)


@pytest.fixture(autouse=True)
def spans_off() -> None:
    """Every test starts from the disabled default; each opts into `enable_spans()`."""
    disable_spans()


def test_disabled_by_default() -> None:
    """The module switch starts off, matching `spans_enabled()`."""
    assert spans_enabled() is False


def test_enable_and_disable_toggle_the_switch() -> None:
    enable_spans()
    assert spans_enabled() is True
    disable_spans()
    assert spans_enabled() is False


def test_disabled_span_records_nothing() -> None:
    """A disabled span still works as a context manager but writes no record."""
    with span("idle"):
        pass
    assert default_collector().records() == []


def test_enabled_span_records_wall_time() -> None:
    enable_spans()
    with span("load"):
        time.sleep(0.001)
    records = default_collector().records()
    assert len(records) == 1
    assert records[0].name == "load"
    assert records[0].path == "load"
    assert records[0].depth == 0
    assert records[0].wall_ms > 0.0


def test_nesting_builds_dotted_path_and_depth() -> None:
    enable_spans()
    with span("pipeline"), span("extract"):
        pass
    records = default_collector().records()
    assert [(r.name, r.path, r.depth) for r in records] == [
        ("extract", "pipeline.extract", 1),
        ("pipeline", "pipeline", 0),
    ]


def test_span_pops_stack_on_exception() -> None:
    """The stack unwinds even when the block raises, and the span still records."""
    enable_spans()
    with pytest.raises(ValueError, match="boom"), span("risky"):
        raise ValueError("boom")
    assert span_module._stack.get() == ()
    records = default_collector().records()
    assert records[0].name == "risky"


def test_stack_is_empty_after_top_level_span_closes() -> None:
    enable_spans()
    with span("a"):
        with span("b"):
            assert span_module._stack.get() == ("a", "b")
        assert span_module._stack.get() == ("a",)
    assert span_module._stack.get() == ()


def test_bare_decorator_uses_qualname() -> None:
    enable_spans()

    @span
    def compute() -> int:
        return 42

    assert compute() == 42
    records = default_collector().records()
    assert records[0].name == "test_bare_decorator_uses_qualname.<locals>.compute"


def test_named_decorator_uses_given_label() -> None:
    enable_spans()

    @span("custom-label")
    def compute() -> int:
        return 7

    assert compute() == 7
    assert default_collector().records()[0].name == "custom-label"


def test_decorator_disabled_records_nothing_but_still_calls_through() -> None:
    @span("noop")
    def compute() -> int:
        return 1

    assert compute() == 1
    assert default_collector().records() == []


def test_decorator_concurrent_calls_do_not_share_span_state() -> None:
    """Each call opens its own `Span`, so overlapping calls from different threads
    never corrupt each other's timing or stack token."""
    enable_spans()
    collector = Collector()
    ready = threading.Barrier(8)

    @span("worker", collector=collector)
    def work() -> None:
        ready.wait(timeout=5)

    threads = [threading.Thread(target=work) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)

    records = collector.records()
    assert len(records) == 8
    assert all(r.name == "worker" and r.depth == 0 for r in records)


def test_memory_false_leaves_deltas_none() -> None:
    enable_spans()
    with span("x", memory=False):
        pass
    record = default_collector().records()[0]
    assert record.rss_delta_bytes is None
    assert record.gpu_delta_bytes is None


def test_memory_true_records_process_rss_delta() -> None:
    enable_spans()
    with span("x", memory=True):
        bytearray(1024 * 1024)  # nudge RSS so the delta is plausible, sign unconstrained
    record = default_collector().records()[0]
    assert isinstance(record.rss_delta_bytes, int)
    assert isinstance(record.gpu_delta_bytes, int)


def test_memory_true_sums_gpu_bytes_across_detected_gpus(idle_gpu_host: object) -> None:
    enable_spans()
    assert span_module._gpu_used_bytes() == 100  # FakeGPU: total=1000, used_pct=10.0
    with span("x", memory=True):
        pass
    record = default_collector().records()[0]
    assert record.gpu_delta_bytes == 0  # nothing changed the fake GPU between enter/exit


def test_custom_collector_is_isolated_from_default() -> None:
    enable_spans()
    own = Collector()
    with span("scoped", collector=own):
        pass
    assert len(own.records()) == 1
    assert default_collector().records() == []


def test_span_object_reused_directly_still_isolates_state() -> None:
    """Even entering the *same* `Span` instance twice in sequence behaves correctly
    (each `__enter__`/`__exit__` pair is self-contained)."""
    enable_spans()
    s = Span("reused")
    with s:
        pass
    with s:
        pass
    assert len(default_collector().records()) == 2


def test_exit_without_enter_is_a_noop() -> None:
    """Calling `__exit__` on a `Span` that was never entered records nothing."""
    enable_spans()
    s = Span("never-entered")
    s.__exit__(None, None, None)
    assert default_collector().records() == []


def test_collector_reset_clears_log_and_stats() -> None:
    enable_spans()
    collector = Collector()
    with span("x", collector=collector):
        pass
    assert collector.records()
    assert collector.stats()
    collector.reset()
    assert collector.records() == []
    assert collector.stats() == []


def test_collector_records_returns_a_copy() -> None:
    enable_spans()
    collector = Collector()
    with span("x", collector=collector):
        pass
    snapshot = collector.records()
    snapshot.clear()
    assert len(collector.records()) == 1


def test_collector_report_empty() -> None:
    assert Collector().report() == "No spans recorded."


def test_collector_report_and_str_agree() -> None:
    enable_spans()
    collector = Collector()
    with span("x", collector=collector):
        pass
    assert str(collector) == collector.report()
    assert "x" in collector.report()


def test_collector_show_smoke(capsys: pytest.CaptureFixture[str]) -> None:
    """`show()` prints something for both the empty and populated cases."""
    Collector().show(color=False)
    assert "No spans recorded" in capsys.readouterr().out

    enable_spans()
    collector = Collector()
    with span("x", collector=collector):
        pass
    collector.show(color=False)
    assert "x" in capsys.readouterr().out


def test_stats_are_sorted_by_total_time_descending() -> None:
    enable_spans()
    collector = Collector()
    with span("slow", collector=collector):
        time.sleep(0.002)
    with span("fast", collector=collector):
        pass
    stats = collector.stats()
    assert [s.path for s in stats] == ["slow", "fast"]


@given(values=st.lists(wall_times, min_size=1, max_size=50))
def test_collector_aggregation_matches_manual_reduction(values: list[float]) -> None:
    """`count`/`total_ms`/`mean_ms`/`max_ms` are exact reductions over the raw values,
    independent of the bounded reservoir used for the quantiles."""
    collector = Collector(reservoir_size=2048)
    for value in values:
        collector.add(SpanRecord(name="x", path="x", depth=0, wall_ms=value))
    (stat,) = collector.stats()
    assert stat.count == len(values)
    assert math.isclose(stat.total_ms, sum(values), rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(stat.mean_ms, sum(values) / len(values), rel_tol=1e-9, abs_tol=1e-9)
    assert stat.max_ms == max(values)


@given(values=st.lists(wall_times, min_size=1, max_size=50))
def test_reservoir_quantiles_exact_within_capacity(values: list[float]) -> None:
    """With `len(values) <= capacity`, Algorithm R keeps every value, so `quantile`
    matches a plain sort-and-index reference implementation exactly."""
    reservoir = Reservoir(capacity=2048)
    for value in values:
        reservoir.add(value)
    ordered = sorted(values)
    for q in (0.0, 0.5, 0.95, 1.0):
        expected = ordered[min(len(ordered) - 1, int(q * len(ordered)))]
        assert reservoir.quantile(q) == expected


class _FixedRandom(Random):
    """A `random.Random` whose `randint` always returns a fixed value, so a test can
    force Algorithm R's replace/skip branch deterministically."""

    def __init__(self, value: int) -> None:
        super().__init__()
        self.value = value

    def randint(self, a: int, b: int) -> int:
        return self.value


def test_reservoir_replaces_a_slot_when_selected_within_capacity() -> None:
    """Once full, a `slot < capacity` draw overwrites that slot with the new value."""
    reservoir = Reservoir(capacity=2)
    reservoir.add(1.0)
    reservoir.add(2.0)
    reservoir.random = _FixedRandom(0)
    reservoir.add(99.0)
    assert reservoir.values == [99.0, 2.0]


def test_reservoir_skips_replacement_when_slot_lands_outside_capacity() -> None:
    """A `slot >= capacity` draw leaves every kept value untouched."""
    reservoir = Reservoir(capacity=2)
    reservoir.add(1.0)
    reservoir.add(2.0)
    reservoir.random = _FixedRandom(5)
    reservoir.add(99.0)
    assert reservoir.values == [1.0, 2.0]


@given(values=st.lists(wall_times, min_size=1, max_size=500))
def test_reservoir_quantiles_are_monotonic_and_bounded(values: list[float]) -> None:
    """p50 <= p95 <= max always holds, and every quantile lies within the value range,
    whether or not the stream exceeded the reservoir's capacity."""
    reservoir = Reservoir(capacity=16)
    for value in values:
        reservoir.add(value)
    p50, p95 = reservoir.quantile(0.50), reservoir.quantile(0.95)
    assert min(values) <= p50 <= p95 <= max(values)


@given(values=st.lists(wall_times, min_size=0, max_size=5))
def test_reservoir_empty_or_small_quantile(values: list[float]) -> None:
    """An empty reservoir reads 0.0; a small one still returns one of its own values."""
    reservoir = Reservoir(capacity=16)
    for value in values:
        reservoir.add(value)
    if not values:
        assert reservoir.quantile(0.5) == 0.0
    else:
        assert reservoir.quantile(0.5) in values


def _tree() -> st.SearchStrategy[tuple[str, list]]:
    return st.recursive(
        st.tuples(names, st.just([])),
        lambda children: st.tuples(names, st.lists(children, max_size=2)),
        max_leaves=12,
    )


def _run_tree(node: tuple[str, list], collector: Collector) -> None:
    label, children = node
    with span(label, collector=collector):
        for child in children:
            _run_tree(child, collector)


def _expected_paths(node: tuple[str, list], prefix: tuple[str, ...]) -> list[tuple[str, int]]:
    label, children = node
    path = ".".join((*prefix, label))
    depth = len(prefix)
    results: list[tuple[str, int]] = []
    for child in children:
        results += _expected_paths(child, (*prefix, label))
    results.append((path, depth))
    return results


@given(tree=_tree())
def test_nesting_invariant_matches_reference_walk(tree: tuple[str, list]) -> None:
    """For any span tree, the recorded (path, depth) sequence — in completion order —
    matches a post-order reference walk, and the stack is empty once it all closes."""
    enable_spans()
    collector = Collector()
    _run_tree(tree, collector)
    actual = [(r.path, r.depth) for r in collector.records()]
    assert actual == _expected_paths(tree, ())
    assert span_module._stack.get() == ()
