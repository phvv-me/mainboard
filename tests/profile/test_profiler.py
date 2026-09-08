import importlib
import inspect
import time
from collections.abc import Sequence
from typing import NoReturn

import pytest

from mainboard import Collection, Profiler, span
from mainboard.probe import GPU
from mainboard.profile import (
    Activity,
    BottleneckReport,
    DeviceEvidence,
    Feature,
    Tracer,
    annotate,
)

from .support import FakeGPU, FakeSnapshot, clock_tracer, one_process_gpu

# `mainboard.profile` shadows the `benchmark` submodule with its re-exported function as an
# attribute lookup; `import_module` reads `sys.modules` by dotted name, unaffected.
benchmark_module = importlib.import_module("mainboard.profile.benchmark")


def test_profiler_signature_resolves_runtime_annotations() -> None:
    """Public constructor annotations remain available to runtime introspection."""
    signature = inspect.signature(Profiler)
    assert signature.parameters["auto"].annotation == Sequence[str]


def test_collection_features_preserve_supported_wire_values() -> None:
    """Recorded feature bits keep their meaning; the unused collector is not accepted."""
    assert {feature.name: feature.value for feature in Feature} == {
        "SPANS": 2,
        "DEVICE": 4,
        "MARKERS": 8,
        "ACTIVITY": 16,
    }
    assert Feature.DEFAULT.value == 30
    with pytest.raises(ValueError):
        Collection.model_validate({"features": 1})


def test_a_session_records_spans_windows_and_every_aggregate_read_off_them(
    one_gpu: FakeGPU,
) -> None:
    """One bracketed span becomes a summary, a device window, and every derived report.

    `stats`, `bottlenecks`, `trace_report` and `report` are all views of the same frozen
    result, so a session exposes them without a second collection pass.
    """
    with (
        Profiler(
            gpus=(one_gpu,),
            features=Profiler.Feature.SPANS | Profiler.Feature.ACTIVITY,
            activities=Activity.KERNEL,
            sample_interval_ms=1,
        ) as profiler,
        span("step"),
    ):
        assert isinstance(profiler.bottlenecks(), list)
        assert isinstance(profiler.stats(), list)
        assert isinstance(profiler.trace_report(), BottleneckReport)
        assert isinstance(profiler.report(), str)

    result = profiler.result()
    assert [item.name for item in result.summaries] == ["step"]
    assert [window.name for window in result.windows] == ["step"]
    assert any(stat.name == "step" for stat in result.stats())


def test_profiler_exit_without_open_frame_is_safe(one_gpu: FakeGPU) -> None:
    """Closing a region that was never opened is ignored rather than erroring."""
    profiler = Profiler(gpus=(one_gpu,))
    with profiler:
        profiler.exit(999, wall_ns=1)
    assert profiler.result().summaries == ()


def test_short_region_still_records_memory_via_boundary_snapshot(one_gpu: FakeGPU) -> None:
    """A region too fast for the async sampler keeps a boundary snapshot, not a zero peak.

    With a 1-second sampling interval the poller never ticks inside the region, so without
    the synchronous boundary read the memory footprint would be lost, which is the failure
    mode when profiling a fast kernel against a per-call sync barrier.
    """
    with Profiler(gpus=(one_gpu,), sample_interval_ms=1000) as profiler:
        token = profiler.enter("kernel")
        profiler.exit(token, wall_ns=1)
    summary = profiler.result().summaries[0]
    assert summary.samples == 1  # the boundary snapshot, since no async tick landed
    assert summary.peak_memory_bytes == 40  # from the one_gpu fixture snapshot


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


def test_the_sampler_skips_a_failed_read_and_attributes_the_rest(
    one_gpu: FakeGPU, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One sensor read that raises is logged and skipped, and the poll after it still lands.

    The samples reach the open span, so the device and its memory footprint come back on
    the summary rather than being lost with the failed read.
    """
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
    summary = profiler.result().summaries[0]
    assert summary.samples >= 1
    assert summary.peak_memory_bytes == 40
    assert profiler.result().device == "probe"


def test_the_sampler_reads_nothing_without_an_open_span_or_a_device(
    one_gpu: FakeGPU, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A poll with no span open records nothing, and no device means nothing to read."""
    profiler = Profiler(features=Profiler.Feature.DEVICE, sample_interval_ms=1)
    profiler.gpu = one_gpu
    waits = iter((False, True))
    monkeypatch.setattr(profiler.stop_event, "wait", lambda interval: next(waits))
    profiler.sample()
    assert profiler.result().summaries == ()
    assert Profiler(features=Profiler.Feature.DEVICE).target_snapshot("x") is None


def test_markers_are_emitted_only_when_selected(
    one_gpu: FakeGPU, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A span pushes and pops a native range only under MARKERS, and never becomes a summary.

    A marker-only session annotates the native timeline without collecting span timings, so
    the result stays empty while the pushes and pops still pair up.
    """
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

    with Profiler(gpus=(one_gpu,), features=Profiler.Feature.MARKERS) as marker_only, span("m"):
        pass
    assert marker_only.result().summaries == ()


def test_auto_uses_local_monitoring_and_disables_on_exit() -> None:
    """Auto-annotating a module instruments its own functions and releases them on exit."""
    with Profiler(
        features=Profiler.Feature.SPANS,
        auto=("mainboard.profile.benchmark",),
    ) as profiler:
        benchmark_module.benchmark(lambda: None, iters=1, warmup=0)
    assert any("benchmark" in item.name for item in profiler.result().summaries)
    assert annotate.enabled_codes() == ()


def test_activity_window_buffer_is_bounded(one_gpu: FakeGPU) -> None:
    """The device-window buffer is bounded like the span buffer, counting what it drops."""
    with Profiler(
        gpus=(one_gpu,),
        features=Profiler.Feature.SPANS | Profiler.Feature.ACTIVITY,
        max_spans=1,
    ) as profiler:
        with span("one"):
            pass
        with span("two"):
            pass
    assert len(profiler.result().windows) == 1
    assert profiler.result().dropped_spans == 2


def test_the_session_takes_its_device_from_the_collection_policy() -> None:
    """A policy is handed over whole, and it alone decides whether a GPU is selected at all.

    With neither DEVICE nor ACTIVITY requested no GPU is ever touched.
    """
    gpu = one_process_gpu()
    built = Profiler.under(Collection(features=Feature.SPANS), gpus=(gpu,))
    assert built.collection.features is Feature.SPANS
    assert built.gpus == (gpu,)
    with built as profiler:
        assert profiler.gpu is None

    with Profiler(
        gpus=(gpu,), features=Profiler.Feature.DEVICE, device_index=0, sample_interval_ms=1000
    ) as profiler:
        assert profiler.gpu is gpu


def test_an_invalid_device_index_never_profiles_a_different_card() -> None:
    """A misspelled device must not produce evidence labeled with another card."""
    profiler = Profiler(gpus=(one_process_gpu(),), features=Feature.DEVICE, device_index=5)
    with pytest.raises(ValueError, match="device_index 5.*1 visible devices"), profiler:
        pytest.fail("work ran with an invalid device selection")
    assert not profiler.active
    assert profiler.sampler is None


@pytest.mark.parametrize(
    "settings",
    [
        {"device_index": -1},
        {"sample_interval_ms": 0},
        {"sample_interval_ms": -1},
        {"max_spans": 0},
        {"max_spans": -1},
    ],
)
def test_invalid_collection_bounds_fail_before_starting(settings: dict[str, int]) -> None:
    """Device indices and buffer/sampler bounds cannot silently disable collection."""
    with pytest.raises(ValueError):
        Collection.model_validate(settings)


def test_a_session_handed_no_device_profiles_the_hosts_own(
    one_gpu: FakeGPU, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bare `Profiler()` collects device evidence, because the probe already knows the cards.

    This is the whole default. Before, a session nobody handed a probe to attached to no
    device and collected nothing, which reads exactly like a run that did no GPU work.
    """
    monkeypatch.setattr(GPU, "all", staticmethod(lambda: (one_gpu,)))
    with Profiler(sample_interval_ms=1000) as profiler, span("work"):
        pass
    assert profiler.gpus == (one_gpu,)
    assert profiler.gpu is one_gpu
    result = profiler.result()
    assert result.device_evidence is DeviceEvidence.COLLECTED
    assert result.device == "probe"


def test_a_named_device_is_never_second_guessed_by_discovery(
    one_gpu: FakeGPU, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Naming a device settles it, so discovery never runs and never overrides the choice."""

    def unreachable() -> NoReturn:
        raise AssertionError("host discovery ran even though the caller named a device")

    monkeypatch.setattr(GPU, "all", staticmethod(unreachable))
    with Profiler(gpus=(one_gpu,), sample_interval_ms=1000) as profiler:
        assert profiler.gpu is one_gpu


def test_activity_asked_for_by_name_with_no_visible_device_is_refused() -> None:
    """A deep GPU trace where there is no GPU refuses, rather than tracing nothing.

    An empty activity pass cannot be told apart from a run that did no GPU work, so the
    session names the situation and the command that shows what the host really has.
    """
    features = Profiler.Feature.SPANS | Profiler.Feature.ACTIVITY
    with pytest.raises(RuntimeError, match="no device is visible"), Profiler(features=features):
        pass  # pragma: no cover  reason=__enter__ refuses before the body runs since=2026-08-27


@pytest.mark.parametrize(
    ("features", "evidence", "said"),
    [
        pytest.param(Feature.DEFAULT, DeviceEvidence.ABSENT, True, id="asked_and_got_none"),
        pytest.param(Feature.SPANS, DeviceEvidence.UNSOUGHT, False, id="never_asked"),
    ],
)
def test_a_deviceless_session_says_which_kind_of_silence_it_is(
    features: Feature, evidence: DeviceEvidence, said: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A GPU-less host still profiles its Python, and the result admits the GPU half is empty.

    Under `DEFAULT` the caller asked for everything worth having, so no card is not an error,
    it is a result that has to say which half of it is missing. A spans-only session was
    never promised device evidence, so its silence needs no note.
    """
    monkeypatch.setattr(annotate, "_tracer", clock_tracer())
    with Profiler(features=features) as profiler, span("work"):
        pass
    result = profiler.result()
    assert profiler.gpu is None
    assert result.device_evidence is evidence
    assert ("no device evidence collected" in result.report()) is said
