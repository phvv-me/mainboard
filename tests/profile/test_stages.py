from collections.abc import Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING

import pytest

from mainboard import Profiler as RealProfiler
from mainboard.probe import GPU
from mainboard.profile import (
    Activity,
    BenchSample,
    Profile,
    RegionSummary,
    StageProfile,
    Tracer,
    annotate,
    profile_stages,
    stages,
)

from .support import FakeGPU

if TYPE_CHECKING:
    from collections.abc import Iterator
    from types import TracebackType

    from mainboard.profile.protocols import DeviceProbe


def _profile_with_region() -> Profile:
    return Profile(device="fake", summaries=(RegionSummary(name="r", wall_ms=1.0),))


def test_profile_stages_benchmarks_each_case_without_requesting_a_trace() -> None:
    calls = {"a": 0, "b": 0}

    def bump(key: str) -> None:
        calls[key] += 1

    result = profile_stages({"a": lambda: bump("a"), "b": lambda: bump("b")}, iters=3, warmup=1)
    assert isinstance(result, StageProfile)
    assert [s.label for s in result.samples] == ["a", "b"]
    assert all(isinstance(s, BenchSample) and s.runs == 3 for s in result.samples)
    assert calls == {"a": 4, "b": 4}  # (warmup 1 + iters 3) per stage
    assert result.profile is None


def test_a_requested_trace_without_a_visible_gpu_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(GPU, "all", staticmethod(lambda: ()))
    with pytest.raises(RuntimeError, match="GPU activity collection was requested"):
        profile_stages({"a": lambda: None}, trace=True, iters=1, warmup=0)


def test_a_visible_gpu_without_an_activity_collector_is_not_a_trace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Detecting hardware does not prove that CUPTI or another activity backend works."""
    monkeypatch.setattr(annotate, "_tracer", Tracer())
    with pytest.raises(ValueError, match="no activity collector available"):
        profile_stages({"a": lambda: None}, gpu=FakeGPU(), trace=True, iters=1, warmup=0)


def test_the_timing_table_lists_every_stage_or_says_there_were_none(
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = profile_stages({"step": lambda: None}, iters=2, warmup=0)
    text = str(result)
    assert "stage" in text and "step" in text and "mean" in text
    result.show()
    assert "step" in capsys.readouterr().out
    assert StageProfile().timing_text() == "No stages profiled."


def test_a_traced_stage_profile_appends_the_deep_report(
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = StageProfile(
        samples=(BenchSample(label="r", samples=(1.0,)),), profile=_profile_with_region()
    )
    text = str(result)
    assert "stage" in text and "Spans" in text
    result.show()
    assert "r" in capsys.readouterr().out


class _StubProfiler:
    """A CUDA-free `Profiler` recording what it was opened with."""

    Feature = RealProfiler.Feature
    opened_with: Activity | None = None
    devices: Sequence[DeviceProbe] = ()

    def __init__(
        self, *, gpus: Sequence[DeviceProbe], features: Feature, activities: Activity
    ) -> None:
        del features
        _StubProfiler.devices = gpus
        _StubProfiler.opened_with = activities

    def __enter__(self) -> _StubProfiler:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        return None

    def result(self) -> Profile:
        return _profile_with_region()


@pytest.mark.parametrize(
    ("trace", "opened_with", "barrier"),
    [(Activity.KERNEL, Activity.KERNEL, True), (True, Activity.ALL, False)],
    ids=["exactly_these_kinds_drained_by_a_barrier", "everything_the_device_offers"],
)
def test_profile_stages_runs_one_trace_pass_when_a_gpu_is_present(
    monkeypatch: pytest.MonkeyPatch, trace: bool | Activity, opened_with: Activity, barrier: bool
) -> None:
    monkeypatch.setattr(stages, "Profiler", _StubProfiler)
    ran: list[str] = []
    synced: list[int] = []

    result = profile_stages(
        {"a": lambda: ran.append("a"), "b": lambda: ran.append("b")},
        gpu=FakeGPU(),
        sync=(lambda: synced.append(1)) if barrier else None,
        trace=trace,
        iters=1,
        warmup=0,
    )
    assert ran[-2:] == ["a", "b"]  # each case ran inside its span during the trace pass
    assert len(synced) == (6 if barrier else 0)  # two stages benchmarked, then traced
    assert _StubProfiler.opened_with is opened_with
    assert result.profile is not None and result.profile.device == "fake"


def test_a_stage_drains_inside_its_attribution_window(monkeypatch: pytest.MonkeyPatch) -> None:
    """The device barrier belongs before the span closes, not after its end timestamp."""
    events: list[str] = []

    @contextmanager
    def window(name: str) -> Iterator[None]:
        events.append(f"open:{name}")
        yield
        events.append(f"close:{name}")

    monkeypatch.setattr(stages, "Profiler", _StubProfiler)
    monkeypatch.setattr(stages, "span", window)
    profile_stages(
        {"a": lambda: events.append("work")},
        trace=True,
        sync=lambda: events.append("drain"),
        iters=1,
        warmup=0,
    )
    assert events[-4:] == ["open:a", "work", "drain", "close:a"]
    assert _StubProfiler.devices == ()  # an omitted probe uses Profiler's discovery
