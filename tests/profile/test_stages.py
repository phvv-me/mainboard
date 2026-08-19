"""`profile_stages`/`StageProfile`: benchmark a set of named steps, optionally traced."""

from typing import TYPE_CHECKING

from mainboard import Profiler as RealProfiler
from mainboard.profile import (
    Activity,
    BenchSample,
    Feature,
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

    import pytest

    from mainboard.profile.protocols import DeviceProbe


def test_profile_stages_benchmarks_each_case_without_trace() -> None:
    """Each named stage becomes one `BenchSample`; no GPU means no deep trace."""
    calls = {"a": 0, "b": 0}

    def bump(key: str) -> None:
        calls[key] += 1

    result = profile_stages(
        {"a": lambda: bump("a"), "b": lambda: bump("b")},
        iters=3,
        warmup=1,
    )
    assert isinstance(result, StageProfile)
    assert [s.label for s in result.samples] == ["a", "b"]
    assert all(isinstance(s, BenchSample) and s.runs == 3 for s in result.samples)
    assert calls == {"a": 4, "b": 4}  # (warmup 1 + iters 3) per stage
    assert result.profile is None


def test_profile_stages_skips_trace_without_a_gpu() -> None:
    """`trace=True` is silently skipped when no GPU is given, so the call still works."""
    result = profile_stages({"x": lambda: None}, trace=True, iters=2, warmup=0)
    assert result.profile is None
    assert result.samples[0].label == "x"


def test_stage_profile_str_shows_timing_table() -> None:
    """The result stringifies into a readable per-stage table."""
    result = profile_stages({"step": lambda: None}, iters=2, warmup=0)
    text = str(result)
    assert "stage" in text and "step" in text and "mean" in text


def test_stage_profile_empty_is_reported() -> None:
    """A profile with no stages says so rather than emitting a blank table."""
    assert StageProfile().timing_text() == "No stages profiled."


def test_stage_profile_show_prints_timing(capsys: pytest.CaptureFixture[str]) -> None:
    """`show` prints the per-stage table when there is no deep trace."""
    profile_stages({"only": lambda: None}, iters=1, warmup=0).show()
    assert "only" in capsys.readouterr().out


def _profile_with_region() -> Profile:
    """A `Profile` carrying one region, built without a GPU for the traced branches."""
    return Profile(device="fake", summaries=(RegionSummary(name="r", wall_ms=1.0),))


def test_stage_profile_str_appends_deep_report_when_traced() -> None:
    """When a trace is present, `__str__` appends the region and trace reports."""
    sample = BenchSample(label="r", samples=(1.0,))
    result = StageProfile(samples=(sample,), profile=_profile_with_region())
    text = str(result)
    assert "stage" in text and "Spans" in text


def test_stage_profile_show_renders_deep_report(capsys: pytest.CaptureFixture[str]) -> None:
    """`show` prints the plain-text profile view when a trace is present."""
    sample = BenchSample(label="r", samples=(1.0,))
    StageProfile(samples=(sample,), profile=_profile_with_region()).show()
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


def test_profile_stages_runs_one_trace_pass_when_gpu_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With a GPU and `trace`, every stage is bracketed and run inside the trace pass."""
    monkeypatch.setattr(stages, "Profiler", _StubProfiler)
    ran: list[str] = []
    synced: list[int] = []

    result = profile_stages(
        {"a": lambda: ran.append("a"), "b": lambda: ran.append("b")},
        gpu=FakeGPU(),
        sync=lambda: synced.append(1),
        trace=Activity.KERNEL,
        iters=1,
        warmup=0,
    )
    assert ran[-2:] == ["a", "b"]  # each case ran inside its span during the trace pass
    assert len(synced) >= 2  # benchmark + trace pass both synced
    assert _StubProfiler.opened_with is Activity.KERNEL
    assert result.profile is not None and result.profile.device == "fake"


def test_profile_stages_trace_true_uses_all_activity_kinds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`trace=True` opens the full `Activity.ALL` kind set."""
    monkeypatch.setattr(stages, "Profiler", _StubProfiler)
    profile_stages({"x": lambda: None}, gpu=FakeGPU(), trace=True, iters=1, warmup=0)
    assert _StubProfiler.opened_with is Activity.ALL
