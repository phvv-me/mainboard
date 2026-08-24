# `profile_stages` and `StageProfile`, which benchmark a set of named steps and optionally
# trace them in one pass.

from typing import TYPE_CHECKING

import pytest

from mainboard import Profiler as RealProfiler
from mainboard.profile import (
    Activity,
    BenchSample,
    Profile,
    RegionSummary,
    StageProfile,
    profile_stages,
    stages,
)

from .conftest import FakeGPU

if TYPE_CHECKING:
    from collections.abc import Sequence
    from types import TracebackType

    from mainboard.profile.protocols import DeviceProbe


def _profile_with_region() -> Profile:
    """A `Profile` carrying one region, built without a GPU for the traced branches."""
    return Profile(device="fake", summaries=(RegionSummary(name="r", wall_ms=1.0),))


def test_profile_stages_benchmarks_each_case_and_skips_a_trace_without_a_gpu() -> None:
    """Each named stage becomes one `BenchSample`, and no GPU means no deep trace.

    `trace=True` is silently skipped rather than refused when no device was given, so the
    same call works on a CPU-only host and on a GPU one.
    """
    calls = {"a": 0, "b": 0}

    def bump(key: str) -> None:
        calls[key] += 1

    result = profile_stages(
        {"a": lambda: bump("a"), "b": lambda: bump("b")}, trace=True, iters=3, warmup=1
    )
    assert isinstance(result, StageProfile)
    assert [s.label for s in result.samples] == ["a", "b"]
    assert all(isinstance(s, BenchSample) and s.runs == 3 for s in result.samples)
    assert calls == {"a": 4, "b": 4}  # (warmup 1 + iters 3) per stage
    assert result.profile is None


def test_the_timing_table_lists_every_stage_or_says_there_were_none(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The result stringifies into a readable per-stage table, and `show` prints it.

    A profile with no stages at all says so rather than emitting a blank table.
    """
    result = profile_stages({"step": lambda: None}, iters=2, warmup=0)
    text = str(result)
    assert "stage" in text and "step" in text and "mean" in text
    result.show()
    assert "step" in capsys.readouterr().out
    assert StageProfile().timing_text() == "No stages profiled."


def test_a_traced_stage_profile_appends_the_deep_report(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """With a trace present, the rendering appends the region and trace reports."""
    result = StageProfile(
        samples=(BenchSample(label="r", samples=(1.0,)),), profile=_profile_with_region()
    )
    text = str(result)
    assert "stage" in text and "Spans" in text
    result.show()
    assert "r" in capsys.readouterr().out


class _StubProfiler:
    """No-op stand-in for the CUPTI `Profiler` so the trace orchestration is testable.

    Records the `Activity` kinds it was opened with and returns a fixed `Profile`,
    standing in for the GPU-only backend that cannot run without CUDA.
    """

    Feature = RealProfiler.Feature
    opened_with: Activity | None = None

    def __init__(
        self, *, gpus: Sequence[DeviceProbe], features: Feature, activities: Activity
    ) -> None:
        del gpus, features
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
    """With a GPU and `trace`, every stage is bracketed and run inside one trace pass.

    `trace=True` asks for every kind the device offers, and a `sync` barrier, when given,
    drains each region before the next one opens.
    """
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
