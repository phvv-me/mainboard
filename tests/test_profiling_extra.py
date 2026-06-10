"""The remaining profiling seams: the CLI `profile` command, the sampling thread, the
PEP 669 auto-annotation stack, snapshot aggregation, and a few resolve/detect branches.

All mocked: the CLI runs a fake module under a fake profiler, the sampler runs against a
fake one-GPU probe, and auto-annotation is driven through the real `sys.monitoring` hooks
with a no-op tracer — so nothing here needs a GPU.
"""

from __future__ import annotations

from typing import Any

import pytest

from mainboard import cli
from mainboard.profiling import annotate
from mainboard.profiling import profiler as profiler_mod
from mainboard.profiling.models import RegionSummary
from mainboard.profiling.profiler import Profiler
from mainboard.profiling.result import Profile
from mainboard.profiling.trace import Activity
from mainboard.profiling.tracer import Tracer


class FakeProfiler:
    """A `Profiler` stand-in capturing how the CLI opened it and what it returned."""

    last: FakeProfiler | None = None

    def __init__(self, *, trace: bool = False, auto: tuple[str, ...] = ()) -> None:
        self.trace = trace
        self.auto = auto
        FakeProfiler.last = self

    def __enter__(self) -> FakeProfiler:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def result(self) -> Profile:
        return Profile(device="cli", summaries=(RegionSummary(name="r", wall_ms=1.0),))


def test_cli_profile_runs_a_module(monkeypatch: pytest.MonkeyPatch) -> None:
    """`profile <module>` runs it under the profiler and shows the result."""
    import runpy

    ran: list[str] = []
    monkeypatch.setattr("mainboard.profiling.Profiler", FakeProfiler)
    monkeypatch.setattr(runpy, "run_module", lambda target, run_name: ran.append(target))
    cli.profile("pkg.mod", auto="a,b", color=False)
    assert ran == ["pkg.mod"]
    assert FakeProfiler.last is not None and FakeProfiler.last.auto == ("a", "b")


def test_cli_profile_runs_a_script_and_exports(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """`profile script.py --perfetto` runs the path and writes the timeline."""
    import runpy

    ran: list[str] = []
    monkeypatch.setattr("mainboard.profiling.Profiler", FakeProfiler)
    monkeypatch.setattr(runpy, "run_path", lambda target, run_name: ran.append(target))
    out = tmp_path / "t.json"
    cli.profile("script.py", trace=True, perfetto=str(out), color=False)
    assert ran == ["script.py"]
    assert out.exists()  # the Profile was exported to Perfetto JSON


# ── Profiler sampling thread + auto-annotation ───────────────────────────────────


class ClockTracer(Tracer):
    """A no-op tracer with a monotonic device clock for region windows."""

    def __init__(self) -> None:
        self._clock = 0

    def timestamp(self) -> int:
        self._clock += 1
        return self._clock


@pytest.fixture
def one_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """A one-GPU host whose snapshot carries fixed telemetry, with a no-op tracer."""
    from mainboard.enums import UnitKind, Vendor
    from mainboard.gpu import GPU
    from mainboard.models.gpu_snapshot import GPUSnapshot
    from mainboard.models.memory import Memory
    from mainboard.models.utilization import Utilization

    class OneGPU(GPU):
        @classmethod
        def all(cls) -> tuple[GPU, ...]:
            return (cls(),)

        def snapshot(self, name: str = "") -> GPUSnapshot:
            return GPUSnapshot(
                name=name,
                unit_name="probe",
                kind=UnitKind.GPU,
                vendor=Vendor.UNKNOWN,
                memory=Memory(total_bytes=100, used_bytes=40),
                utilization=Utilization(gpu_pct=25, memory_pct=10),
            )

    monkeypatch.setattr(profiler_mod.GPU, "all", classmethod(lambda cls: (OneGPU(),)))
    monkeypatch.setattr(annotate, "_tracer", ClockTracer())
    monkeypatch.setattr(annotate, "profiler", None)


def test_profiler_sampler_attributes_snapshots_to_region(one_gpu: None) -> None:
    """The background sampler folds device snapshots into the open region's summary."""
    import time

    with Profiler(sample_interval_ms=1) as profiler, annotate.region("work"):
        time.sleep(0.02)  # let the sampler tick at least once
    summary = next(s for s in profiler.summaries() if s.name == "work")
    assert summary.samples >= 1
    assert summary.peak_memory_bytes == 40
    assert summary.avg_util_pct == 25.0
    assert summary.avg_memory_util_pct == 10.0


def test_profiler_auto_annotates_a_module(one_gpu: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """`auto` instruments a package by file root, and disables on context exit."""
    import importlib

    target_module = importlib.import_module("mainboard.profiling.benchmark")
    pushes: list[str] = []
    monkeypatch.setattr(annotate.tracer(), "push", lambda name: pushes.append(name))

    with Profiler(auto=["mainboard.profiling.benchmark"]) as profiler:
        target_module.benchmark(lambda: None, iters=1, warmup=0)
    # auto-annotation wrapped calls under the package, then tore down on exit
    assert any("benchmark" in name for name in pushes)
    assert isinstance(profiler.result(), Profile)
    assert annotate._predicate is None  # disabled on exit


def test_profiler_module_roots_skips_modules_without_file() -> None:
    """A namespace-ish module with no `__file__` contributes no root rather than failing."""
    import sys
    import types as _types

    fake = _types.ModuleType("mb_fake_no_file")
    sys.modules["mb_fake_no_file"] = fake
    try:
        assert Profiler._module_roots(["mb_fake_no_file"]) == []
    finally:
        del sys.modules["mb_fake_no_file"]


# ── PEP 669 auto-annotation hooks (direct) ───────────────────────────────────────


def test_auto_hooks_balance_returns_and_unwinds(monkeypatch: pytest.MonkeyPatch) -> None:
    """The start/return/unwind hooks push and pop the per-thread frame stack evenly."""
    monkeypatch.setattr(annotate, "_tracer", ClockTracer())
    monkeypatch.setattr(annotate, "profiler", None)
    monkeypatch.setattr(annotate, "_predicate", lambda code: True)
    annotate._stack().clear()
    code = (lambda: None).__code__
    annotate._on_start(code, 0)
    annotate._on_start(code, 0)
    assert len(annotate._stack()) == 2
    annotate._on_return(code, 0, None)
    annotate._on_unwind(code, 0, ValueError())
    assert annotate._stack() == []
    annotate._on_return(code, 0, None)  # empty stack: ignored


def test_auto_start_skips_unmatched_code(monkeypatch: pytest.MonkeyPatch) -> None:
    """A code object failing the predicate is tracked as an empty (None) frame."""
    monkeypatch.setattr(annotate, "_tracer", ClockTracer())
    monkeypatch.setattr(annotate, "_predicate", lambda code: False)
    annotate._stack().clear()
    annotate._on_start((lambda: None).__code__, 0)
    assert annotate._stack() == [None]
    annotate._stack().clear()


# ── resolve / detect / aggregation branches ──────────────────────────────────────


def test_resolve_all_with_full_support_drops_nothing() -> None:
    """`Activity.ALL` against a device supporting everything adapts to itself, dropping none."""

    class FullTracer(Tracer):
        def supported(self) -> Activity:
            return Activity.ALL

    assert FullTracer().resolve(Activity.ALL) == Activity.ALL


def test_detect_returns_present_vendor_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """`detect` prefers a backend whose vendor matches a GPU actually present."""
    from mainboard.enums import Vendor
    from mainboard.gpu import GPU

    class PresentTracer(Tracer):
        vendor = Vendor.NVIDIA

        @classmethod
        def is_available(cls) -> bool:
            return True

    class FakeGPU:
        vendor = Vendor.NVIDIA

    monkeypatch.setattr(Tracer, "registry", classmethod(lambda cls: [Tracer, PresentTracer]))
    monkeypatch.setattr(GPU, "all", classmethod(lambda cls: (FakeGPU(),)))
    assert type(Tracer.detect()) is PresentTracer


def test_detect_falls_back_to_any_available_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """With an available backend but no matching present GPU, `detect` takes the first."""
    from mainboard.enums import Vendor
    from mainboard.gpu import GPU

    class AnyTracer(Tracer):
        vendor = Vendor.AMD

        @classmethod
        def is_available(cls) -> bool:
            return True

    monkeypatch.setattr(Tracer, "registry", classmethod(lambda cls: [Tracer, AnyTracer]))
    monkeypatch.setattr(GPU, "all", classmethod(lambda cls: ()))
    assert type(Tracer.detect()) is AnyTracer


def test_region_summary_from_empty_snaps_is_wall_only() -> None:
    """A region that sampled nothing keeps its wall time and zeroed telemetry."""
    summary = RegionSummary.from_snaps("r", 5.0, [])
    assert summary.wall_ms == 5.0
    assert summary.samples == 0


def test_stack_initializes_per_thread() -> None:
    """The auto-annotation frame stack is created lazily and is thread-local."""
    import threading

    seen: list[int] = []
    thread = threading.Thread(target=lambda: seen.append(len(annotate._stack())))
    thread.start()
    thread.join()
    assert seen == [0]  # a fresh thread starts with an empty stack


def test_auto_hooks_exit_an_active_region(monkeypatch: pytest.MonkeyPatch) -> None:
    """When a profiler is active, the return hook closes the region the start hook opened."""
    closed: list[str] = []

    class ActiveProfiler:
        def enter(self, name: str) -> None:
            pass

        def exit(self, name: str, wall_ns: int) -> None:
            closed.append(name)

    monkeypatch.setattr(annotate, "_tracer", ClockTracer())
    monkeypatch.setattr(annotate, "profiler", ActiveProfiler())
    monkeypatch.setattr(annotate, "_predicate", lambda code: True)
    annotate._stack().clear()
    code = (lambda: None).__code__
    annotate._on_start(code, 0)
    annotate._on_return(code, 0, None)
    assert closed == [code.co_qualname]


def test_profiler_exit_without_enter_is_safe() -> None:
    """Exiting a profiler that was never entered joins no thread and clears no auto."""
    Profiler().__exit__()  # no thread started -> the join branch is skipped


def test_sampler_skips_when_no_region_is_open(
    one_gpu: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A sampler tick with no open region takes no snapshot and simply continues."""
    profiler = Profiler(sample_interval_ms=1)
    profiler._gpu = profiler_mod.GPU.all()[0]
    waits = iter([False, True])  # one live tick (no frames), then stop
    monkeypatch.setattr(profiler._stop, "wait", lambda _interval: next(waits))
    profiler._sample()  # the live tick sees no frames and continues to the stop
    assert profiler.summaries() == []


def test_hot_region_attributes_kernel_to_narrowest_window() -> None:
    """A kernel inside nested windows is attributed to the tightest enclosing region."""
    from mainboard.profiling.trace import KernelTrace, RegionWindow

    profile = Profile(
        windows=(
            RegionWindow(name="inner", start_ns=100, end_ns=300, wall_ns=200),
            RegionWindow(name="outer", start_ns=0, end_ns=1000, wall_ns=1000),
        ),
        kernels=(KernelTrace(name="k", start_ns=150, end_ns=200),),
    )
    report = profile.trace_report()
    # inner is seen first and kept; the wider outer matches too but does not displace it
    assert report.hot_regions[0].name == "inner"
