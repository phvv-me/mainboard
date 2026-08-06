import importlib
import subprocess
import sys
import threading
import time
import types
from pathlib import Path
from typing import Any

import pytest

from mainboard import cli
from mainboard.gpu import GPU
from mainboard.models.gpu_snapshot import GPUSnapshot
from mainboard.profiling import Profiler, annotate, span
from mainboard.profiling.models import RegionSummary
from mainboard.profiling.profiler import Collection, Reach
from mainboard.profiling.python import PythonAction, PythonProfile, Tachyon
from mainboard.profiling.result import Profile
from mainboard.profiling.trace import KernelTrace, RegionWindow
from mainboard.profiling.tracer import Tracer


class FakeProfiler:
    """Capture CLI requests without launching a target."""

    Feature = Profiler.Feature
    target = ""
    options: dict[str, Any] = {}

    @classmethod
    def run(cls, target: str, **options: Any) -> Profile:
        cls.target = target
        cls.options = options
        return Profile(summaries=(RegionSummary(name="r", wall_ms=1.0),))

    @classmethod
    def attach(cls, pid: int, **options: Any) -> Profile:
        cls.target = str(pid)
        cls.options = options
        return Profile(summaries=(RegionSummary(name="attach", wall_ms=1.0),))

    @classmethod
    def dump(cls, pid: int, **options: Any) -> Profile:
        cls.target = str(pid)
        cls.options = options
        return Profile(summaries=(RegionSummary(name="dump", wall_ms=1.0),))


def test_cli_composes_collector_features(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "Profiler", FakeProfiler)
    cli.profile_run(
        "pkg.mod",
        python=False,
        device=False,
        markers=False,
        activity=False,
        color=False,
    )
    assert FakeProfiler.target == "pkg.mod"
    assert FakeProfiler.options["features"] == Profiler.Feature.SPANS


def test_cli_forwards_python_output(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(cli, "Profiler", FakeProfiler)
    output = tmp_path / "profile.html"
    cli.profile_run("script.py", output=str(output), color=False)
    # The flag has to reach the sampler model the CLI builds, which is the whole point of
    # building it at the boundary rather than forwarding eight loose arguments.
    assert FakeProfiler.options["sampler"].output == output


def test_cli_attach_and_dump_share_the_profiler(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "Profiler", FakeProfiler)
    cli.attach(10, color=False)
    assert FakeProfiler.target == "10"
    cli.dump(20, color=False)
    assert FakeProfiler.target == "20"


def test_sampler_attributes_target_process_snapshots(one_gpu: object) -> None:
    with Profiler(sample_interval_ms=1) as profiler, span("work"):
        time.sleep(0.02)
    summary = profiler.result().summaries[0]
    assert summary.samples >= 1
    assert summary.peak_memory_bytes == 40
    assert profiler.result().device == "probe"


def test_detected_but_unused_gpu_is_absent(
    one_gpu: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    gpu = profiler_gpu(one_gpu)
    monkeypatch.setattr(type(gpu), "snapshot", lambda self, name="": GPUSnapshot(name=name))
    with Profiler(sample_interval_ms=1) as profiler, span("cpu"):
        pass
    result = profiler.result()
    assert result.device == ""
    assert result.summaries[0].samples == 0


def profiler_gpu(one_gpu: object) -> GPU:
    """Return the concrete fake device selected by `Profiler`."""
    del one_gpu
    return GPU.all()[0]


def test_sampler_skips_failed_reads(one_gpu: object, monkeypatch: pytest.MonkeyPatch) -> None:
    gpu = profiler_gpu(one_gpu)
    calls = 0
    original = gpu.snapshot

    def flaky(self: GPU, name: str = "") -> GPUSnapshot:
        del self
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("sensor")
        return original(name)

    monkeypatch.setattr(type(gpu), "snapshot", flaky)
    with Profiler(sample_interval_ms=1) as profiler, span("work"):
        time.sleep(0.01)
    assert profiler.result().summaries[0].samples >= 1


def test_sampler_with_no_open_span_reads_nothing(
    one_gpu: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    profiler = Profiler(features=Profiler.Feature.DEVICE, sample_interval_ms=1)
    profiler.gpu = profiler_gpu(one_gpu)
    waits = iter((False, True))
    monkeypatch.setattr(profiler.stop_event, "wait", lambda interval: next(waits))
    profiler.sample()
    assert profiler.result().summaries == ()


def test_target_snapshot_without_a_selected_gpu_is_empty() -> None:
    assert Profiler(features=Profiler.Feature.DEVICE).target_snapshot("x") is None


def test_markers_are_emitted_only_when_selected(
    one_gpu: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    del one_gpu
    pushes: list[str] = []
    pops: list[bool] = []
    tracer = Tracer()
    monkeypatch.setattr(tracer, "push", pushes.append)
    monkeypatch.setattr(tracer, "pop", lambda: pops.append(True))
    monkeypatch.setattr(annotate, "_tracer", tracer)
    with (
        Profiler(features=Profiler.Feature.SPANS | Profiler.Feature.MARKERS) as profiler,
        span("marked"),
    ):
        pass
    assert pushes == ["marked"]
    assert pops == [True]
    assert profiler.result().summaries


def test_marker_only_session_does_not_create_span_results(
    one_gpu: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    del one_gpu
    monkeypatch.setattr(annotate, "_tracer", Tracer())
    with Profiler(features=Profiler.Feature.MARKERS) as profiler, span("marker"):
        pass
    assert profiler.result().summaries == ()


def test_auto_uses_local_monitoring_and_disables_on_exit() -> None:
    benchmark_module = importlib.import_module("mainboard.profiling.benchmark")

    with Profiler(
        features=Profiler.Feature.SPANS,
        auto=("mainboard.profiling.benchmark",),
    ) as profiler:
        benchmark_module.benchmark(lambda: None, iters=1, warmup=0)
    assert any("benchmark" in item.name for item in profiler.result().summaries)
    assert annotate._codes == ()


def test_module_codes_handles_missing_file(monkeypatch: pytest.MonkeyPatch) -> None:
    module = types.ModuleType("mainboard_fake_module")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    assert Profiler.module_codes((module.__name__,)) == set()


def test_module_codes_excludes_imported_functions(monkeypatch: pytest.MonkeyPatch) -> None:
    module = types.ModuleType("mainboard_owned_module")

    def owned() -> None:
        pass

    def imported() -> None:
        pass

    owned.__module__ = module.__name__
    vars(module).update(owned=owned, imported=imported)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    assert Profiler.module_codes((module.__name__,)) == {owned.__code__}


def test_monitor_hooks_balance_return_unwind_and_empty_stack() -> None:
    code = (lambda: None).__code__
    annotate.frames().clear()
    annotate.on_start(code, 0)
    annotate.on_start(code, 0)
    annotate.on_return(code, 0, None)
    annotate.on_unwind(code, 0, ValueError())
    annotate.on_return(code, 0, None)
    annotate.on_return(code, 0, None)
    assert annotate.frames() == []


def test_monitor_unwind_closes_only_selected_code() -> None:
    selected = (lambda: None).__code__
    other = (lambda: 1).__code__
    annotate._codes = (selected,)
    annotate.frames().clear()
    annotate.on_start(selected, 0)
    annotate.on_unwind(other, 0, ValueError())
    assert len(annotate.frames()) == 1
    annotate.on_unwind(selected, 0, ValueError())
    assert annotate.frames() == []
    annotate._codes = ()


def test_monitor_stack_is_thread_local() -> None:
    seen: list[int] = []
    thread = threading.Thread(target=lambda: seen.append(len(annotate.frames())))
    thread.start()
    thread.join(timeout=5)
    assert seen == [0]


def test_run_without_python_sampling_executes_target_once(tmp_path: Path) -> None:
    marker = tmp_path / "ran"
    script = tmp_path / "target.py"
    script.write_text(f"from pathlib import Path\nPath({str(marker)!r}).write_text('yes')\n")
    profile = Profiler.run(
        str(script),
        features=Profiler.Feature.SPANS,
    )
    assert marker.read_text() == "yes"
    assert [item.name for item in profile.summaries] == ["program"]


def test_instrumented_run_uses_the_selected_interpreter(tmp_path: Path) -> None:
    # `Target.launch` starts `executable` as a bare subprocess, and a venv's own python is
    # itself a symlink (uv's `.venv/bin/python3` -> `.venv/bin/python` -> the real interpreter).
    # CPython locates `pyvenv.cfg` by walking up from the *invoked* path's own directory, not
    # the fully resolved target, so a companion symlink only reaches the venv's dependencies
    # (pydantic, here) when it sits beside the real executable. A symlink under `tmp_path` loses
    # the venv, and the target subprocess dies with `ModuleNotFoundError` instead of running.
    executable = Path(sys.executable).with_name("mainboard-test-selected-python")
    executable.unlink(missing_ok=True)
    executable.symlink_to(sys.executable)
    try:
        marker = tmp_path / "interpreter"
        script = tmp_path / "target.py"
        script.write_text(
            "import sys\n"
            "from pathlib import Path\n"
            f"Path({str(marker)!r}).write_text(sys.executable)\n"
        )
        profile = Profiler.run(
            str(script),
            features=Profiler.Feature.SPANS,
            sampler=Tachyon(executable=executable),
        )
        assert Path(marker.read_text()) == executable
        assert [item.name for item in profile.summaries] == ["program"]
    finally:
        executable.unlink(missing_ok=True)


def test_instrumented_run_enforces_timeout(tmp_path: Path) -> None:
    script = tmp_path / "target.py"
    script.write_text("import time\ntime.sleep(1)\n")
    with pytest.raises(subprocess.TimeoutExpired):
        Profiler.run(
            str(script),
            features=Profiler.Feature.SPANS,
            timeout=0.01,
        )


def test_strict_python_sampling_fails_before_target(tmp_path: Path) -> None:
    script = tmp_path / "target.py"
    script.write_text("raise AssertionError('must not run')\n")
    with pytest.raises(RuntimeError, match="Python sampling requires"):
        Profiler.run(str(script), features=Profiler.Feature.PYTHON, strict=True)


def test_python_only_fallback_runs_target_once(tmp_path: Path) -> None:
    marker = tmp_path / "count"
    script = tmp_path / "target.py"
    script.write_text(f"from pathlib import Path\nPath({str(marker)!r}).write_text('1')\n")
    assert Profiler.run(str(script), features=Profiler.Feature.PYTHON) == Profile()
    assert marker.read_text() == "1"


def fake_python_run(
    self: Tachyon,
    target: str,
    *,
    module: bool | None = None,
    args: tuple[str, ...] = (),
    timeout: float | None = None,
    import_paths: tuple[Path, ...] = (),
) -> PythonProfile:
    """Stand in for Tachyon and emit the child profile artifact when requested."""
    del self, module, timeout, import_paths
    if target == "mainboard.profiling.runner":
        Profile(summaries=(RegionSummary(name="program", wall_ms=1.0),)).save(args[3])
    return PythonProfile(
        action=PythonAction.RUN, command=(target,), target=target, stdout="python"
    )


def test_python_sampling_combines_with_local_collectors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Tachyon, "available", lambda self: True)
    monkeypatch.setattr(Tachyon, "run", fake_python_run)
    profile = Profiler.run("pkg.mod", features=Profiler.Feature.PYTHON | Profiler.Feature.SPANS)
    assert profile.python is not None
    assert profile.summaries[0].name == "program"


def test_python_only_sampling_uses_the_direct_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Tachyon, "available", lambda self: True)
    monkeypatch.setattr(Tachyon, "run", fake_python_run)
    profile = Profiler.run("pkg.mod", features=Profiler.Feature.PYTHON)
    assert profile.python is not None
    assert profile.python.target == "pkg.mod"


def test_public_sampler_forwards_consistent_worker_thread_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sampled: list[Tachyon] = []

    def capture(
        self: Tachyon,
        target: str,
        *,
        module: bool | None = None,
        args: tuple[str, ...] = (),
        timeout: float | None = None,
        import_paths: tuple[Path, ...] = (),
    ) -> PythonProfile:
        del module, args, timeout, import_paths
        sampled.append(self)
        return PythonProfile(action=PythonAction.RUN, command=(target,), target=target)

    monkeypatch.setattr(Tachyon, "available", lambda self: True)
    monkeypatch.setattr(Tachyon, "run", capture)
    Profiler.run(
        "pkg.mod",
        features=Profiler.Feature.PYTHON,
        sampler=Tachyon(all_threads=True, blocking=True),
    )
    assert sampled[0].all_threads is True
    assert sampled[0].blocking is True


def test_attach_and_dump_wrap_private_tachyon(monkeypatch: pytest.MonkeyPatch) -> None:
    result = PythonProfile(action=PythonAction.ATTACH, command=("attach",), stdout="ok")
    attached: list[Tachyon] = []
    monkeypatch.setattr(Tachyon, "available", lambda self: True)

    def capture_attach(self: Tachyon, pid: int, timeout: float | None = None) -> PythonProfile:
        del pid, timeout
        attached.append(self)
        return result

    monkeypatch.setattr(Tachyon, "attach", capture_attach)
    monkeypatch.setattr(
        Tachyon,
        "dump",
        lambda self, pid, timeout=None: result.model_copy(
            update={"action": PythonAction.DUMP, "command": ("dump",)}
        ),
    )
    sampler = Tachyon(all_threads=True, blocking=True)
    assert Profiler.attach(1, sampler=sampler).python is result
    assert attached[0].all_threads is True
    assert attached[0].blocking is True
    assert Profiler.dump(1).python is not None


def test_attach_and_dump_reject_unsupported_python(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Tachyon, "available", lambda self: False)
    with pytest.raises(RuntimeError, match="Python sampling requires Python 3.15"):
        Profiler.attach(1)
    with pytest.raises(RuntimeError, match="Python sampling requires Python 3.15"):
        Profiler.dump(1)


def test_reach_kind_reports_which_of_the_three_ways_it_names() -> None:
    assert Reach.here().kind == "here"
    assert Reach.launch("pkg.mod").kind == "launch"
    assert Reach.attaching(123).kind == "attach"


def test_measure_dispatches_an_attach_reach_to_the_python_sampler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = PythonProfile(action=PythonAction.ATTACH, command=("attach",), stdout="ok")
    monkeypatch.setattr(Tachyon, "available", lambda self: True)
    monkeypatch.setattr(Tachyon, "attach", lambda self, pid, timeout=None: result)
    profile = Profiler.measure(Reach.attaching(123))
    assert profile.python is result


def test_measure_dispatches_a_launch_reach_to_run(tmp_path: Path) -> None:
    marker = tmp_path / "ran"
    script = tmp_path / "target.py"
    script.write_text(f"from pathlib import Path\nPath({str(marker)!r}).write_text('yes')\n")
    profile = Profiler.measure(
        Reach.launch(str(script)), collection=Collection(features=Profiler.Feature.SPANS)
    )
    assert marker.read_text() == "yes"
    assert [item.name for item in profile.summaries] == ["program"]


def test_measure_rejects_a_here_reach_since_it_has_no_target_to_launch() -> None:
    with pytest.raises(ValueError, match="cannot launch"):
        Profiler.measure(Reach.here())


def test_profile_result_omits_empty_lanes_and_reports_gpu_activity() -> None:
    assert Profile().report() == "No profiling data collected."
    profile = Profile(kernels=(KernelTrace(name="k", start_ns=0, end_ns=1_000),))
    assert "GPU activity" in profile.report()
    assert "Python" not in profile.report()


def test_python_and_capture_limit_render_only_when_present(
    capsys: pytest.CaptureFixture[str],
) -> None:
    python = PythonProfile(action=PythonAction.RUN, command=("run",), stdout="sample")
    profile = Profile(python=python, dropped_spans=2, dropped_activities=3)
    assert "Python profile" in profile.report()
    assert "oldest GPU activities dropped" in profile.report()
    profile.show(color=False)
    output = capsys.readouterr().out
    assert "sample" in output
    assert "oldest spans dropped" in output
    assert "oldest GPU activities dropped" in output
    Profile(python=python.model_copy(update={"stdout": ""})).show(color=False)
    assert "No text output" in capsys.readouterr().out


def test_activity_window_buffer_is_bounded(cpu_only_host: None) -> None:
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


def test_hot_region_uses_the_narrowest_window() -> None:
    profile = Profile(
        windows=(
            RegionWindow(name="inner", start_ns=100, end_ns=300, wall_ns=200),
            RegionWindow(name="outer", start_ns=0, end_ns=1000, wall_ns=1000),
        ),
        kernels=(KernelTrace(name="k", start_ns=150, end_ns=200),),
    )
    assert profile.trace_report().hot_regions[0].name == "inner"
