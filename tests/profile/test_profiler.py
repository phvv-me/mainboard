import importlib
import inspect
import time
from collections.abc import Sequence

import pytest

from mainboard import Collection, Profiler, Reach, span
from mainboard.profile import Activity, BottleneckReport, Feature, Tracer, annotate

from .conftest import FakeGPU, FakeSnapshot, one_process_gpu

# `mainboard.profile` shadows the `benchmark` submodule with its re-exported function as an
# attribute lookup; `import_module` reads `sys.modules` by dotted name, unaffected.
benchmark_module = importlib.import_module("mainboard.profile.benchmark")


def test_profiler_signature_resolves_runtime_annotations() -> None:
    """Public constructor annotations remain available to runtime introspection."""
    signature = inspect.signature(Profiler)
    assert signature.parameters["auto"].annotation == Sequence[str]


def test_profiler_times_regions_and_aggregates(one_gpu: FakeGPU) -> None:
    """The profiler brackets a span, samples the target GPU, and yields a stat."""
    with Profiler(gpus=(one_gpu,), sample_interval_ms=1) as profiler, span("step"):
        pass
    stats = profiler.result().stats()
    assert any(s.name == "step" for s in stats)
    assert profiler.result().summaries[0].name == "step"


def test_profiler_deep_trace_opens_collector(one_gpu: FakeGPU) -> None:
    """The activity feature opens a collector and records span windows."""
    with (
        Profiler(
            gpus=(one_gpu,),
            features=Profiler.Feature.SPANS | Profiler.Feature.ACTIVITY,
            activities=Activity.KERNEL,
            sample_interval_ms=1,
        ) as profiler,
        span("k"),
    ):
        pass
    result = profiler.result()
    assert result.windows and result.windows[0].name == "k"


def test_profiler_exit_without_open_frame_is_safe(one_gpu: FakeGPU) -> None:
    """Closing a region that was never opened is ignored rather than erroring."""
    profiler = Profiler(gpus=(one_gpu,))
    with profiler:
        profiler.exit(999, wall_ns=1)
    assert profiler.result().summaries == ()


def test_short_region_still_records_memory_via_boundary_snapshot(one_gpu: FakeGPU) -> None:
    """A region too fast for the async sampler keeps a boundary snapshot, not a zero peak.

    With a 1-second sampling interval the poller never ticks inside the region, so without
    the synchronous boundary read the memory footprint would be lost — the failure mode
    when profiling a fast kernel against a per-call sync barrier.
    """
    with Profiler(gpus=(one_gpu,), sample_interval_ms=1000) as profiler:
        token = profiler.enter("kernel")
        profiler.exit(token, wall_ns=1)
    summary = profiler.result().summaries[0]
    assert summary.samples == 1  # the boundary snapshot, since no async tick landed
    assert summary.peak_memory_bytes == 40  # from the one_gpu fixture snapshot


def test_profiler_trace_report_and_report(one_gpu: FakeGPU) -> None:
    """The profiler proxies the result's stats/bottlenecks/trace_report/report."""
    with Profiler(
        gpus=(one_gpu,),
        features=Profiler.Feature.SPANS | Profiler.Feature.ACTIVITY,
        activities=Activity.KERNEL,
        sample_interval_ms=1,
    ) as profiler:
        with span("k"):
            pass
        assert isinstance(profiler.bottlenecks(), list)
        assert isinstance(profiler.stats(), list)
        assert isinstance(profiler.trace_report(), BottleneckReport)
        assert isinstance(profiler.report(), str)


def test_sampler_attributes_target_process_snapshots(one_gpu: FakeGPU) -> None:
    with Profiler(gpus=(one_gpu,), sample_interval_ms=1) as profiler, span("work"):
        time.sleep(0.02)
    summary = profiler.result().summaries[0]
    assert summary.samples >= 1
    assert summary.peak_memory_bytes == 40
    assert profiler.result().device == "probe"


def test_detected_but_unused_gpu_is_absent(
    one_gpu: FakeGPU, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A GPU with no process match on it never surfaces in the result."""
    monkeypatch.setattr(one_gpu, "reading", FakeSnapshot())
    with Profiler(gpus=(one_gpu,), sample_interval_ms=1) as profiler, span("cpu"):
        pass
    result = profiler.result()
    assert result.device == ""
    assert result.summaries[0].samples == 0


def test_sampler_skips_failed_reads(one_gpu: FakeGPU, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0
    original = one_gpu.snapshot

    def flaky(name: str = "") -> FakeSnapshot:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("sensor")
        return original(name)

    monkeypatch.setattr(one_gpu, "snapshot", flaky)
    # Wait for two reads rather than a fixed sleep, since Windows' ~15ms timer resolution
    # can deliver only one tick for a 1ms interval.
    with Profiler(gpus=(one_gpu,), sample_interval_ms=1) as profiler, span("work"):
        deadline = time.monotonic() + 5.0
        while calls < 2 and time.monotonic() < deadline:
            time.sleep(0.005)
    assert calls >= 2
    assert profiler.result().summaries[0].samples >= 1


def test_sampler_with_no_open_span_reads_nothing(
    one_gpu: FakeGPU, monkeypatch: pytest.MonkeyPatch
) -> None:
    profiler = Profiler(features=Profiler.Feature.DEVICE, sample_interval_ms=1)
    profiler.gpu = one_gpu
    waits = iter((False, True))
    monkeypatch.setattr(profiler.stop_event, "wait", lambda interval: next(waits))
    profiler.sample()
    assert profiler.result().summaries == ()


def test_target_snapshot_without_a_selected_gpu_is_empty() -> None:
    assert Profiler(features=Profiler.Feature.DEVICE).target_snapshot("x") is None


def test_markers_are_emitted_only_when_selected(
    one_gpu: FakeGPU, monkeypatch: pytest.MonkeyPatch
) -> None:
    pushes: list[str] = []
    pops: list[bool] = []
    tracer = Tracer()
    monkeypatch.setattr(tracer, "push", pushes.append)
    monkeypatch.setattr(tracer, "pop", lambda: pops.append(True))
    monkeypatch.setattr(annotate, "_tracer", tracer)
    with (
        Profiler(
            gpus=(one_gpu,), features=Profiler.Feature.SPANS | Profiler.Feature.MARKERS
        ) as profiler,
        span("marked"),
    ):
        pass
    assert pushes == ["marked"]
    assert pops == [True]
    assert profiler.result().summaries


def test_marker_only_session_does_not_create_span_results(
    one_gpu: FakeGPU, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(annotate, "_tracer", Tracer())
    with Profiler(gpus=(one_gpu,), features=Profiler.Feature.MARKERS) as profiler, span("marker"):
        pass
    assert profiler.result().summaries == ()


def test_auto_uses_local_monitoring_and_disables_on_exit() -> None:
    with Profiler(
        features=Profiler.Feature.SPANS,
        auto=("mainboard.profile.benchmark",),
    ) as profiler:
        benchmark_module.benchmark(lambda: None, iters=1, warmup=0)
    assert any("benchmark" in item.name for item in profiler.result().summaries)
    assert annotate.enabled_codes() == ()


def test_activity_window_buffer_is_bounded() -> None:
    with Profiler(
        features=Profiler.Feature.SPANS | Profiler.Feature.ACTIVITY,
        max_spans=1,
    ) as profiler:
        with span("one"):
            pass
        with span("two"):
            pass
    assert len(profiler.result().windows) == 1
    assert profiler.result().dropped_spans == 2


def test_profiler_under_builds_from_a_collection_and_gpus() -> None:
    collection = Collection(features=Feature.SPANS)
    gpu = one_process_gpu()
    profiler = Profiler.under(collection, gpus=(gpu,))
    assert profiler.collection.features is Feature.SPANS
    assert profiler.gpus == (gpu,)


def test_no_gpus_selected_when_device_and_activity_off() -> None:
    """With neither DEVICE nor ACTIVITY requested, no GPU is ever selected."""
    gpu = one_process_gpu()
    with Profiler(gpus=(gpu,), features=Profiler.Feature.SPANS) as profiler:
        assert profiler.gpu is None


def test_device_index_out_of_range_falls_back_to_first_gpu() -> None:
    gpu = one_process_gpu()
    with Profiler(
        gpus=(gpu,), features=Profiler.Feature.DEVICE, device_index=5, sample_interval_ms=1000
    ) as profiler:
        assert profiler.gpu is gpu


def test_reach_kind_reports_which_of_the_three_ways_it_names() -> None:
    assert Reach.here().kind == "here"
    assert Reach.launch("pkg.mod").kind == "launch"
    assert Reach.attaching(123).kind == "attach"


def test_reach_launch_and_attaching_carry_their_fields() -> None:
    launch = Reach.launch("pkg.mod", module=True, args=("--flag",), timeout=5.0)
    assert launch.target == "pkg.mod"
    assert launch.module is True
    assert launch.args == ("--flag",)
    assert launch.timeout == 5.0

    attach = Reach.attaching(123, timeout=1.0)
    assert attach.pid == 123
    assert attach.timeout == 1.0
