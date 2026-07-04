"""The remaining profiling seams: the CLI `profile` command, the sampling thread, the
PEP 669 auto-annotation stack, snapshot aggregation, and a few resolve/detect branches.

All mocked: the CLI runs a fake module under a fake profiler, the sampler runs against a
fake one-GPU probe, and auto-annotation is driven through the real `sys.monitoring` hooks
with a no-op tracer — so nothing here needs a GPU.
"""

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

from .conftest import make_clock_tracer


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


def test_profiler_sampler_attributes_snapshots_to_region(one_gpu: object) -> None:
    """The background sampler folds device snapshots into the open region's summary."""
    import time

    with Profiler(sample_interval_ms=1) as profiler, annotate.region("work"):
        time.sleep(0.02)  # let the sampler tick at least once
    summary = next(s for s in profiler.summaries() if s.name == "work")
    assert summary.samples >= 1
    assert summary.peak_memory_bytes == 40
    assert summary.avg_util_pct == 25.0
    assert summary.avg_memory_util_pct == 10.0


def test_profiler_auto_annotates_a_module(
    one_gpu: object, monkeypatch: pytest.MonkeyPatch
) -> None:
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
    monkeypatch.setattr(annotate, "_tracer", make_clock_tracer())
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
    monkeypatch.setattr(annotate, "_tracer", make_clock_tracer())
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

    monkeypatch.setattr(annotate, "_tracer", make_clock_tracer())
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
    one_gpu: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A sampler tick with no open region takes no snapshot and simply continues."""
    profiler = Profiler(sample_interval_ms=1)
    profiler._gpu = profiler_mod.GPU.all()[0]
    waits = iter([False, True])  # one live tick (no frames), then stop
    monkeypatch.setattr(profiler._stop, "wait", lambda _interval: next(waits))
    profiler._sample()  # the live tick sees no frames and continues to the stop
    assert profiler.summaries() == []


def test_sampler_survives_a_snapshot_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """A transient snapshot failure skips that tick instead of killing the sampler."""
    from mainboard.models.gpu_snapshot import GPUSnapshot

    class FlakyGPU:
        def __init__(self) -> None:
            self.calls = 0

        def snapshot(self, name: str = "") -> GPUSnapshot:
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("nvml hiccup")
            return GPUSnapshot(name=name)

    profiler = Profiler(sample_interval_ms=1)
    profiler._gpu = FlakyGPU()  # pyrefly: ignore[bad-assignment]
    profiler.enter("region")
    waits = iter([False, False, True])  # a failing tick, a good tick, then stop
    monkeypatch.setattr(profiler._stop, "wait", lambda _interval: next(waits))
    profiler._sample()
    profiler.exit("region", wall_ns=1)
    assert profiler.summaries()[0].samples == 1  # only the good tick landed


def test_boundary_snapshot_failure_is_logged_not_raised() -> None:
    """A region too short for the sampler that also fails its own boundary read logs and continues.

    The synchronous fallback in `exit` is best-effort: a transient sensor failure at the
    boundary must not surface to the caller closing the region, only cost that region its
    memory reading.
    """
    from mainboard.models.gpu_snapshot import GPUSnapshot

    class FailingGPU:
        def snapshot(self, name: str = "") -> GPUSnapshot:
            raise RuntimeError("nvml hiccup")

    profiler = Profiler(sample_interval_ms=1000)
    profiler._gpu = FailingGPU()  # pyrefly: ignore[bad-assignment]
    profiler.enter("kernel")
    profiler.exit("kernel", wall_ns=1)  # no async tick landed, then the boundary read also fails
    assert profiler.summaries()[0].samples == 0


def test_exit_closes_the_calling_threads_frame() -> None:
    """A worker's exit closes its own frame even when the main thread opened a later one."""
    import threading

    profiler = Profiler()
    worker_open = threading.Event()
    main_open = threading.Event()

    def work() -> None:
        profiler.enter("worker")
        worker_open.set()
        assert main_open.wait(timeout=5)
        profiler.exit("worker", wall_ns=1)

    thread = threading.Thread(target=work)
    thread.start()
    assert worker_open.wait(timeout=5)
    profiler.enter("main")
    main_open.set()
    thread.join(timeout=5)
    # the worker's exit must not have stolen the main thread's still-open frame
    assert [frame.name for frame in profiler._frames] == ["main"]
    profiler.exit("main", wall_ns=1)
    assert [summary.name for summary in profiler.summaries()] == ["worker", "main"]


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
