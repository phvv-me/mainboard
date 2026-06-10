"""The vendor-neutral profiling core and the CUPTI/NVTX backend, all mocked.

Covers the result value (`Profile` and its verbs), the rich/plain renderers, the
Perfetto export, the sampling `Profiler`, the annotation surface, and the NVIDIA
CUPTI/NVTX tracer — none of which needs a real GPU here: the device clock, the CUPTI
activity stream, and the NVTX/ROCTx/signpost libraries are all faked.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from mainboard.profiling import annotate, perfetto, render
from mainboard.profiling.benchmark import compare
from mainboard.profiling.models import RegionStat, RegionSummary
from mainboard.profiling.profiler import Profiler
from mainboard.profiling.result import Profile, ProfileDiff
from mainboard.profiling.trace import (
    Activity,
    ActivityRecord,
    BottleneckReport,
    CallbackSession,
    KernelTrace,
    MemcpyTrace,
    RegionWindow,
    TraceCollector,
)
from mainboard.profiling.tracer import Tracer


def _traced_profile() -> Profile:
    """A profile with two regions and kernels/memcpys binned across their windows."""
    return Profile(
        device="dev",
        summaries=(
            RegionSummary(name="encode", wall_ms=2.0, avg_util_pct=50.0, avg_power_w=100.0),
            RegionSummary(name="decode", wall_ms=1.0),
        ),
        windows=(
            RegionWindow(name="encode", start_ns=0, end_ns=1000, wall_ns=2_000_000),
            RegionWindow(name="decode", start_ns=1000, end_ns=2000, wall_ns=1_000_000),
        ),
        kernels=(
            KernelTrace(name="gemm", start_ns=0, end_ns=600, grid="8x1x1", block="256x1x1"),
            KernelTrace(name="gemm", start_ns=1000, end_ns=1400),
            KernelTrace(name="relu", start_ns=600, end_ns=700),
        ),
        memcpys=(MemcpyTrace(kind="HtoD", start_ns=0, end_ns=100, bytes_moved=4096),),
        activities=(
            ActivityRecord(kind="runtime", name="cudaLaunchKernel", start_ns=0, end_ns=5),
        ),
    )


# ── Activity / typed records ────────────────────────────────────────────────────


def test_activity_label_falls_back_for_compound_flags() -> None:
    """A single flag labels by name; a compound flag has no name, so labels `activity`."""
    assert Activity.KERNEL.label == "kernel"
    assert (Activity.KERNEL | Activity.MEMCPY).label == "default"
    assert Activity(0).label == "activity"


def test_memcpy_trace_from_activity_maps_kind_and_bandwidth() -> None:
    """A CUPTI MEMCPY record maps copy_kind to a label and yields a bandwidth."""
    act: Any = types.SimpleNamespace(copy_kind=1, start=0, end=1000, bytes=2000)
    trace = MemcpyTrace.from_activity(act)
    assert trace.kind == "HtoD"
    assert trace.bandwidth_gbps == 2.0
    blank: Any = types.SimpleNamespace(copy_kind=99, start=0, end=0)
    unknown = MemcpyTrace.from_activity(blank)
    assert unknown.kind == "kind_99"
    assert unknown.bandwidth_gbps == 0.0  # zero duration


def test_kernel_trace_from_activity_reads_launch_shape() -> None:
    """A CUPTI CONCURRENT_KERNEL record fills the launch shape from snake_case attrs."""
    act: Any = types.SimpleNamespace(
        name="k",
        start=0,
        end=1000,
        grid_x=8,
        grid_y=1,
        grid_z=1,
        block_x=128,
        block_y=2,
        block_z=1,
        static_shared_memory=512,
        dynamic_shared_memory=256,
        registers_per_thread=40,
    )
    trace = KernelTrace.from_activity(act)
    assert trace.grid == "8x1x1"
    assert trace.block == "128x2x1"
    assert trace.shared_mem == 768
    assert trace.duration_us == 1.0


def test_activity_record_duration() -> None:
    """A generic activity record reports its span."""
    assert ActivityRecord(start_ns=10, end_ns=60).duration_ns == 50


# ── BottleneckReport ─────────────────────────────────────────────────────────────


def test_bottleneck_report_splits_compute_and_copy() -> None:
    """The deep report splits GPU time into compute vs copy and ranks hot spots."""
    report = _traced_profile().trace_report()
    assert isinstance(report, BottleneckReport)
    assert report.compute_pct > report.memcpy_pct
    assert report.hot_kernels[0].name == "gemm"
    assert report.hot_regions[0].name in {"encode", "decode"}


def test_bottleneck_report_attributes_kernel_outside_any_window() -> None:
    """A kernel landing in no region window is labeled rather than dropped."""
    profile = Profile(
        windows=(RegionWindow(name="r", start_ns=0, end_ns=10, wall_ns=10),),
        kernels=(KernelTrace(name="stray", start_ns=100, end_ns=200),),
    )
    report = profile.trace_report()
    assert report.hot_regions[0].name == "(outside regions)"


def test_bottleneck_report_empty_profile() -> None:
    """No traces yields zero totals and empty rankings without dividing by zero."""
    report = Profile().trace_report()
    assert report.total_kernel_ns == 0
    assert report.hot_kernels == ()


# ── TraceCollector / CallbackSession bases ───────────────────────────────────────


def test_trace_collector_base_is_a_noop_context() -> None:
    """The base collector is an empty, safe no-op context manager."""
    with TraceCollector() as collector:
        collector.flush()
        collector.reset()
    assert collector.kernels() == []
    assert collector.memcpys() == []
    assert collector.activities() == []


def test_callback_session_base_counts_nothing() -> None:
    """The base callback session is a no-op that observes no calls."""
    with CallbackSession() as session:
        pass
    assert session.counts() == {}


# ── Tracer base ──────────────────────────────────────────────────────────────────


def test_tracer_base_is_an_unavailable_noop() -> None:
    """The base tracer annotates nothing, supports nothing, and is never available."""
    tracer = Tracer()
    assert Tracer.is_available() is False
    tracer.push("x")
    tracer.pop()
    tracer.mark("x")
    assert tracer.supported() == Activity(0)
    assert isinstance(tracer.timestamp(), int)
    assert isinstance(tracer.collect(), TraceCollector)
    assert isinstance(tracer.callbacks(), CallbackSession)


def test_tracer_resolve_passes_through_when_unsupported() -> None:
    """With no device support, resolve trusts the caller rather than dropping kinds."""
    assert Tracer().resolve(Activity.KERNEL) == Activity.KERNEL


class _SupportingTracer(Tracer):
    """A tracer reporting a fixed support set, to exercise `resolve`."""

    def supported(self) -> Activity:
        return Activity.KERNEL | Activity.MEMCPY


def test_tracer_resolve_adapts_all_to_supported() -> None:
    """`Activity.ALL` adapts down to exactly what the device supports."""
    assert _SupportingTracer().resolve(Activity.ALL) == (Activity.KERNEL | Activity.MEMCPY)


def test_tracer_resolve_fails_fast_on_explicit_unsupported_kind() -> None:
    """An explicit unsupported kind is an error, not a silent omission."""
    with pytest.raises(ValueError, match="not supported"):
        _SupportingTracer().resolve(Activity.MEMORY)


def test_tracer_detect_prefers_present_vendor(monkeypatch: pytest.MonkeyPatch) -> None:
    """`detect` returns the no-op base when no backend library is available."""
    monkeypatch.setattr(Tracer, "registry", classmethod(lambda cls: [Tracer]))
    assert type(Tracer.detect()) is Tracer


# ── Profile result verbs ─────────────────────────────────────────────────────────


def test_profile_stats_and_bottlenecks() -> None:
    """Stats collapse per name; bottlenecks are the slowest by total wall time."""
    profile = _traced_profile()
    stats = profile.stats()
    assert {s.name for s in stats} == {"encode", "decode"}
    assert profile.bottlenecks(top=1)[0].name == "encode"


def test_profile_diff_ranks_regressions(tmp_path: Path) -> None:
    """A diff matches regions by name and ranks by absolute change; round-trips on disk."""
    base = Profile(summaries=(RegionSummary(name="r", wall_ms=2.0),))
    cur = Profile(
        summaries=(RegionSummary(name="r", wall_ms=1.0), RegionSummary(name="new", wall_ms=5.0))
    )
    diff = cur.diff(base)
    assert isinstance(diff, ProfileDiff)
    speedup = next(row for row in diff.rows if row.name == "r").speedup
    assert speedup == 2.0  # 2ms -> 1ms
    path = tmp_path / "p.json"
    cur.save(path)
    assert Profile.load(path).stats()[0].name == "new"


def test_profile_report_and_str_are_plain_text() -> None:
    """`report`/`__str__` give a per-region table; an empty profile says so."""
    assert "encode" in _traced_profile().report()
    assert str(_traced_profile()) == _traced_profile().report()
    assert "No regions" in Profile().report()


def test_profile_perfetto_export(tmp_path: Path) -> None:
    """`Profile.perfetto` writes loadable Chrome trace JSON with all four tracks."""
    path = tmp_path / "trace.json"
    _traced_profile().perfetto(path)
    data = json.loads(path.read_text())
    names = {e["name"] for e in data["traceEvents"]}
    assert {"gemm", "relu", "HtoD", "cudaLaunchKernel"} <= names


def test_perfetto_lays_untraced_regions_sequentially(tmp_path: Path) -> None:
    """Without device windows, regions are placed sequentially by wall time."""
    profile = Profile(
        summaries=(RegionSummary(name="a", wall_ms=1.0), RegionSummary(name="b", wall_ms=2.0))
    )
    path = tmp_path / "t.json"
    perfetto.write_trace(profile, path)
    spans = [e for e in json.loads(path.read_text())["traceEvents"] if e["ph"] == "X"]
    assert [s["name"] for s in spans] == ["a", "b"]
    assert spans[1]["ts"] > spans[0]["ts"]  # b starts after a


def test_perfetto_origin_is_zero_with_no_events(tmp_path: Path) -> None:
    """An empty profile still writes a valid (event-less span) trace."""
    path = tmp_path / "e.json"
    perfetto.write_trace(Profile(), path)
    assert "traceEvents" in json.loads(path.read_text())


# ── Renderers ────────────────────────────────────────────────────────────────────


def test_region_text_handles_empty() -> None:
    """The plain-text region table reports emptiness rather than a blank header."""
    assert render.region_text([]) == "No regions recorded."


def test_show_profile_renders_region_and_kernel_tables(capsys: pytest.CaptureFixture[str]) -> None:
    """`show` prints the region table and, when traced, the hot-kernel table."""
    _traced_profile().show(color=False)
    out = capsys.readouterr().out
    assert "encode" in out and "gemm" in out


def test_profile_rich_renderable_without_kernels() -> None:
    """`__rich__` returns just the region table when there was no deep trace."""
    profile = Profile(summaries=(RegionSummary(name="r", wall_ms=1.0),))
    assert profile.__rich__() is not None


def test_profile_rich_renderable_empty_is_a_message() -> None:
    """An empty profile renders a 'no regions' message rather than a table."""
    assert render.profile_renderable(Profile()) == "No regions recorded."


def test_show_diff_colors_regressions(capsys: pytest.CaptureFixture[str]) -> None:
    """A diff prints baseline/current/delta with a speedup column."""
    base = Profile(summaries=(RegionSummary(name="r", wall_ms=1.0),))
    cur = Profile(summaries=(RegionSummary(name="r", wall_ms=2.0),))
    cur.diff(base).show(color=False)
    assert "r" in capsys.readouterr().out


def test_region_stat_aggregate_collapses_calls() -> None:
    """Repeated occurrences of a name collapse into one stat with the call count."""
    rows = (RegionSummary(name="r", wall_ms=1.0), RegionSummary(name="r", wall_ms=3.0))
    stat = RegionStat.aggregate(rows)[0]
    assert stat.calls == 2
    assert stat.total_ms == 4.0
    assert stat.avg_ms == 2.0


# ── Profiler (sampling, mocked tracer + GPU) ─────────────────────────────────────


class FakeSamplingTracer(Tracer):
    """A tracer with a monotonic device clock and KERNEL/MEMCPY deep support."""

    def __init__(self) -> None:
        self._clock = 0

    def supported(self) -> Activity:
        return Activity.KERNEL | Activity.MEMCPY

    def timestamp(self) -> int:
        self._clock += 1
        return self._clock

    def open(self, kinds: Activity) -> TraceCollector:
        return TraceCollector()


@pytest.fixture
def sampling_host(monkeypatch: pytest.MonkeyPatch) -> None:
    """Wire the profiler's annotate/GPU seam to a fake tracer and a one-GPU probe."""
    from mainboard.gpu import GPU
    from mainboard.models.utilization import Utilization
    from mainboard.profiling import profiler as profiler_mod

    tracer = FakeSamplingTracer()
    monkeypatch.setattr(annotate, "_tracer", tracer)
    monkeypatch.setattr(annotate, "profiler", None)

    class OneGPU(GPU):
        @classmethod
        def all(cls) -> tuple[GPU, ...]:
            return (cls(),)

        @property
        def utilization(self) -> Utilization:
            return Utilization(gpu_pct=30, memory_pct=10)

    monkeypatch.setattr(profiler_mod.GPU, "all", classmethod(lambda cls: (OneGPU(),)))


def test_profiler_times_regions_and_aggregates(sampling_host: None) -> None:
    """The profiler brackets a region, samples the GPU, and yields a stat for it."""
    with Profiler(sample_interval_ms=1) as profiler, annotate.region("step"):
        pass
    stats = profiler.result().stats()
    assert any(s.name == "step" for s in stats)
    assert profiler.summaries()[0].name == "step"


def test_profiler_deep_trace_opens_collector(sampling_host: None) -> None:
    """With `trace`, the profiler opens a collector and records region windows."""
    with Profiler(trace=Activity.KERNEL, sample_interval_ms=1) as profiler, annotate.region("k"):
        pass
    result = profiler.result()
    assert result.windows and result.windows[0].name == "k"
    assert profiler.supported() == (Activity.KERNEL | Activity.MEMCPY)


def test_profiler_exit_without_open_frame_is_safe(sampling_host: None) -> None:
    """Closing a region that was never opened is ignored rather than erroring."""
    profiler = Profiler()
    with profiler:
        profiler.exit("ghost", wall_ns=1)
    assert profiler.summaries() == []


def test_profiler_trace_report_and_show(
    sampling_host: None, capsys: pytest.CaptureFixture[str]
) -> None:
    """The profiler proxies the result's reports and show during/after a run."""
    with Profiler(trace=Activity.KERNEL, sample_interval_ms=1) as profiler:
        with annotate.region("k"):
            pass
        assert isinstance(profiler.bottlenecks(), list)
        assert isinstance(profiler.stats(), list)
        assert isinstance(profiler.trace_report(), BottleneckReport)
        assert isinstance(profiler.report(), str)
        profiler.show(color=False)
    assert capsys.readouterr().out is not None


# ── Annotation surface ───────────────────────────────────────────────────────────


def test_region_without_profiler_only_annotates(monkeypatch: pytest.MonkeyPatch) -> None:
    """A region with no active profiler still pushes/pops native annotation."""
    pushes: list[str] = []
    tracer = FakeSamplingTracer()
    monkeypatch.setattr(tracer, "push", lambda name: pushes.append(name))
    monkeypatch.setattr(annotate, "_tracer", tracer)
    monkeypatch.setattr(annotate, "profiler", None)
    with annotate.region("solo"):
        pass
    assert pushes == ["solo"]


def test_profile_decorator_wraps_each_call(monkeypatch: pytest.MonkeyPatch) -> None:
    """`@profile` names each call a region; bare and parameterized forms both work."""
    monkeypatch.setattr(annotate, "_tracer", FakeSamplingTracer())
    monkeypatch.setattr(annotate, "profiler", None)

    @annotate.profile
    def bare() -> int:
        return 1

    @annotate.profile(name="named")
    def given() -> int:
        return 2

    assert bare() == 1
    assert given() == 2


def test_callbacks_proxies_the_tracer(monkeypatch: pytest.MonkeyPatch) -> None:
    """`callbacks()` returns the active tracer's callback session."""
    monkeypatch.setattr(annotate, "_tracer", FakeSamplingTracer())
    with annotate.callbacks() as session:
        pass
    assert session.counts() == {}


def test_tracer_lazy_detection_is_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    """`annotate.tracer()` detects once and caches the instance."""
    monkeypatch.setattr(annotate, "_tracer", None)
    monkeypatch.setattr(Tracer, "detect", classmethod(lambda cls: Tracer()))
    first = annotate.tracer()
    assert annotate.tracer() is first


def test_enable_auto_instruments_matching_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    """Runtime auto-annotation brackets every call whose code matches the predicate."""
    seen: list[str] = []
    tracer = FakeSamplingTracer()
    monkeypatch.setattr(tracer, "push", lambda name: seen.append(name))
    monkeypatch.setattr(annotate, "_tracer", tracer)
    monkeypatch.setattr(annotate, "profiler", None)

    def target() -> int:
        return 41

    annotate.enable_auto(lambda code: code.co_name == "target")
    try:
        target()
    finally:
        annotate.disable_auto()
    assert any(name.endswith("target") for name in seen)


def test_instrument_source_wraps_function_bodies() -> None:
    """Static AST rewrite wraps each def body in a `region` and prepends the import."""
    rewritten = annotate.instrument_source("def f(x):\n    return x\n")
    assert "_mb_region" in rewritten
    assert "region as _mb_region" in rewritten


# ── benchmark.compare ────────────────────────────────────────────────────────────


def test_compare_tabulates_fastest_first(capsys: pytest.CaptureFixture[str]) -> None:
    """`compare` benchmarks each case and prints them fastest-mean first."""
    samples = compare({"a": lambda: None, "b": lambda: None}, iters=2, warmup=0)
    assert {s.label for s in samples} == {"a", "b"}
    assert capsys.readouterr().out


def _install_fake_module(monkeypatch: pytest.MonkeyPatch, name: str, module: Any) -> None:
    """Register a fake module under ``name`` for the duration of a test."""
    monkeypatch.setitem(sys.modules, name, module)


# ── Vendor annotation backends (ROCTx / signpost) ────────────────────────────────


def test_roctx_tracer_push_pop_mark(monkeypatch: pytest.MonkeyPatch) -> None:
    """The ROCTx backend tracks a range-id stack and forwards marks."""
    from mainboard.providers.amd import tracer as amd_tracer

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
    assert ("start", "r") in events and ("mark", "m") in events


def test_signpost_tracer_intervals(monkeypatch: pytest.MonkeyPatch) -> None:
    """The signpost backend opens/closes intervals via a (name, token) stack."""
    from mainboard.providers.apple import tracer as apple_tracer

    calls: list[tuple[str, Any]] = []

    class FakeSignposter:
        def __init__(self, subsystem: str) -> None:
            calls.append(("init", subsystem))

        def begin_interval(self, name: str) -> str:
            calls.append(("begin", name))
            return f"tok:{name}"

        def end_interval(self, name: str, token: str) -> None:
            calls.append(("end", token))

        def emit_event(self, name: str) -> None:
            calls.append(("event", name))

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
    assert ("begin", "a") in calls and ("event", "e") in calls
