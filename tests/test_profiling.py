"""The reusable profiling helpers: `arch_config` dispatch and `profile_stages`."""

from functools import cached_property

import pytest

from mainboard.gpu import GPU
from mainboard.models.compute_capability import ComputeCapability
from mainboard.profiling import (
    BenchSample,
    Profile,
    Profiler,
    StageProfile,
    arch_config,
    current_arch_key,
    profile_stages,
)
from mainboard.profiling.models import RegionSummary
from mainboard.profiling.trace import Activity


class FakeGPU(GPU):
    """A GPU stand-in that reports a fixed `arch_key` for dispatch tests.

    `all` returns nothing so the registry fan-out never recurses into it; the
    dispatch fixtures inject instances by patching `GPU.all` directly.
    """

    arch_value: str = "sm_90"

    @classmethod
    def all(cls) -> tuple[FakeGPU, ...]:
        return ()

    @cached_property
    def arch_key(self) -> str:
        return self.arch_value


@pytest.fixture
def hopper_host(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pretend `GPU.all()` sees one `sm_90` (Hopper) device."""
    monkeypatch.setattr(GPU, "all", classmethod(lambda cls: (FakeGPU(arch_value="sm_90"),)))


def test_compute_capability_sm_is_dotless_target() -> None:
    """`sm` is the `sm_NN` target string used for per-arch keying."""
    assert ComputeCapability(9, 0).sm == "sm_90"
    assert ComputeCapability(12, 1).sm == "sm_121"


def test_nvidia_arch_key_is_its_compute_capability_target(nvidia_host: object) -> None:
    """An NVIDIA device keys dispatch on its `sm_NN` target (fakes report 8.9)."""
    from mainboard.providers import NvidiaGPU

    assert NvidiaGPU.all()[0].arch_key == "sm_89"


def test_current_arch_key_reads_the_live_gpu(hopper_host: None) -> None:
    """`current_arch_key` returns the first GPU's dispatch key."""
    assert current_arch_key() == "sm_90"


def test_arch_config_selects_the_entry_for_this_gpu(hopper_host: None) -> None:
    """`arch_config` returns the table value keyed on the live GPU's arch."""
    table = {"sm_89": ("ada", 32), "sm_90": ("hopper", 64)}
    assert arch_config(table, default=("cpu", 8)) == ("hopper", 64)


def test_arch_config_falls_back_when_arch_absent(hopper_host: None) -> None:
    """An arch not in the table yields the default rather than raising."""
    assert arch_config({"sm_121": 16}, default=99) == 99


def test_arch_config_defaults_when_no_gpu(cpu_only_host: None) -> None:
    """On a CPU-only host (no GPU) `arch_config` returns the default."""
    assert current_arch_key() is None
    assert arch_config({"sm_90": 1}, default=-1) == -1


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


def test_profile_stages_skips_trace_on_cpu_only_host(cpu_only_host: None) -> None:
    """`trace=True` is silently skipped when no GPU is present, so the call still works."""
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
    sample = BenchSample(label="r", mean_us=1.0, min_us=1.0, runs=1)
    result = StageProfile(samples=(sample,), profile=_profile_with_region())
    text = str(result)
    assert "stage" in text and "region" in text


def test_stage_profile_show_renders_deep_report(capsys: pytest.CaptureFixture[str]) -> None:
    """`show` prints the rich profile view when a trace is present."""
    sample = BenchSample(label="r", mean_us=1.0, min_us=1.0, runs=1)
    StageProfile(samples=(sample,), profile=_profile_with_region()).show(color=False)
    assert "r" in capsys.readouterr().out


class StubProfiler:
    """No-op stand-in for the CUPTI `Profiler` so the trace orchestration is testable.

    Records the `Activity` kinds it was opened with and returns a fixed `Profile`,
    standing in for the GPU-only backend that cannot run without CUDA.
    """

    Feature = Profiler.Feature
    opened_with: Activity | None = None

    def __init__(self, *, features: Profiler.Feature, activities: Activity) -> None:
        del features
        StubProfiler.opened_with = activities

    def __enter__(self) -> StubProfiler:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def result(self) -> Profile:
        return _profile_with_region()


def test_profile_stages_runs_one_trace_pass_when_gpu_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With a GPU and `trace`, every stage is bracketed and run inside the trace pass."""
    from mainboard.profiling import annotate, stages
    from mainboard.profiling.tracer import Tracer

    monkeypatch.setattr(stages.GPU, "all", classmethod(lambda cls: (FakeGPU(),)))
    monkeypatch.setattr(stages, "Profiler", StubProfiler)
    monkeypatch.setattr(annotate, "tracer", Tracer)  # region() reads this no-op tracer
    ran: list[str] = []
    synced: list[int] = []

    result = profile_stages(
        {"a": lambda: ran.append("a"), "b": lambda: ran.append("b")},
        sync=lambda: synced.append(1),
        trace=stages.Activity.KERNEL,
        iters=1,
        warmup=0,
    )
    assert ran[-2:] == ["a", "b"]  # each case ran inside its region during the trace pass
    assert len(synced) >= 2  # benchmark + trace pass both synced
    assert StubProfiler.opened_with is stages.Activity.KERNEL
    assert result.profile is not None and result.profile.device == "fake"


def test_profile_stages_trace_true_uses_all_activity_kinds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`trace=True` opens the full `Activity.ALL` kind set."""
    from mainboard.profiling import annotate, stages
    from mainboard.profiling.tracer import Tracer

    monkeypatch.setattr(stages.GPU, "all", classmethod(lambda cls: (FakeGPU(),)))
    monkeypatch.setattr(stages, "Profiler", StubProfiler)
    monkeypatch.setattr(annotate, "tracer", Tracer)  # region() reads this no-op tracer
    profile_stages({"x": lambda: None}, trace=True, iters=1, warmup=0)
    assert StubProfiler.opened_with is stages.Activity.ALL
