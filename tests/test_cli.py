import json
from pathlib import Path

import pytest

from mainboard import __version__, cli, profiling
from mainboard.profiling.storage import ReadResult, StorageBandwidth


def test_show_renders_via_machine_view(monkeypatch: pytest.MonkeyPatch) -> None:
    """The default command builds a `MachineView` and prints it with the color flag."""
    calls: list[bool] = []

    class FakeView:
        def __init__(self, machine: object) -> None:
            self.machine = machine

        def print(self, *, color: bool = True) -> None:
            calls.append(color)

    monkeypatch.setattr(cli, "MachineView", FakeView)
    cli.show(color=False)
    assert calls == [False]


def test_main_invokes_the_app(monkeypatch: pytest.MonkeyPatch) -> None:
    """`main` runs the cyclopts application object."""
    ran: list[bool] = []
    monkeypatch.setattr(cli, "app", lambda: ran.append(True))
    cli.main()
    assert ran == [True]


def run_cli(*argv: str) -> object:
    """Drive the real cyclopts app with argv, returning its result or exit code.

    Cyclopts returns the command's value (commands here return `None`, so a
    clean run yields `None` or `0`); a `--version`/`--help` flag raises
    `SystemExit`, whose code we surface for the caller to assert on."""
    try:
        return cli.app(list(argv), exit_on_error=False)
    except SystemExit as exit_signal:
        return exit_signal.code if isinstance(exit_signal.code, int) else 1


def ran_clean(result: object) -> bool:
    """Whether `run_cli` returned a success sentinel (a `None`/`0`-returning command)."""
    return result in (None, 0)


def test_version_flag_reports_the_in_tree_version(capsys: pytest.CaptureFixture[str]) -> None:
    """`--version` reports the package's own version, not a stale install's.

    Regression: cyclopts defaults `--version` to `importlib.metadata`, which
    drifts to an older wheel under an editable install. The app pins its version
    to the in-tree `__version__`, so the CLI always names the code it runs."""
    assert run_cli("--version") == 0
    assert capsys.readouterr().out.strip() == __version__


def test_default_command_renders_real_machine_view(
    nvidia_host: object, capsys: pytest.CaptureFixture[str]
) -> None:
    """No-argument dispatch runs the real `MachineView` over a fake NVIDIA host.

    Exercises the whole parse -> default-command -> render path (not a mocked
    view), so a broken `renderable()` or a row reading the GPU telemetry seam
    would surface here rather than slipping past a high mock."""
    assert ran_clean(run_cli())
    out = capsys.readouterr().out
    assert "GPU" in out
    assert "NVIDIA" in out  # the GPU row read the fake telemetry seam
    assert "cuda" in out  # the NVIDIA backend label rendered


def test_no_color_flag_reaches_the_renderer(
    nvidia_host: object, capsys: pytest.CaptureFixture[str]
) -> None:
    """`--no-color` actually disables ANSI styling in the rendered output."""
    assert ran_clean(run_cli("--no-color"))
    assert "\x1b[" not in capsys.readouterr().out


def test_profile_runs_a_script_target_end_to_end(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`profile <script.py>` runs the script as `__main__` under the real profiler.

    Mocks nothing but the filesystem target: the parse, the `runpy` dispatch,
    the `Profiler` context, and `result().show()` all run for real."""
    script = tmp_path / "work.py"
    script.write_text("print('SCRIPT_MARKER', sum(range(100)))\n")
    assert ran_clean(run_cli("profile", str(script)))
    assert "SCRIPT_MARKER 4950" in capsys.readouterr().out


def test_profile_writes_a_perfetto_timeline(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The `--perfetto` path is wired to `result.perfetto`, writing a Chrome trace."""
    script = tmp_path / "work.py"
    script.write_text("x = 1\n")
    timeline = tmp_path / "trace.json"
    assert ran_clean(run_cli("profile", str(script), "--perfetto", str(timeline)))
    payload = json.loads(timeline.read_text())
    assert "traceEvents" in payload


def test_profile_missing_module_surfaces_an_import_error(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A non-existent module target reaches `runpy` and raises, not a silent no-op."""
    with pytest.raises((ImportError, SystemExit)):
        cli.app(["profile", "no.such.module.zzz"], exit_on_error=False)


def test_profile_requires_a_target() -> None:
    """Omitting the required `target` is a usage error the parser rejects.

    Cyclopts raises rather than dispatching `profile` with no target, so the
    command body never runs on a missing argument."""
    from cyclopts.exceptions import CycloptsError

    with pytest.raises(CycloptsError):
        cli.app(["profile"], exit_on_error=False)


def test_storage_command_reports_unavailable(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """An unavailable probe (no CUDA, no scratch, ...) prints the skip reason and returns."""
    monkeypatch.setattr(
        profiling, "nvme_to_hbm", lambda **_: StorageBandwidth(skipped="no CUDA device")
    )
    assert ran_clean(run_cli("storage"))
    assert "unavailable: no CUDA device" in capsys.readouterr().out


def test_storage_command_prints_reads_and_speedup(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A live GDS run prints both reads and the speedup ratio over the mmap bounce."""
    mmap = ReadResult(label="mmap", gigabytes_per_s=5.0, latency_ms=2.0)
    gds = ReadResult(label="gds", gigabytes_per_s=10.0, latency_ms=1.0)
    result = StorageBandwidth(available=True, scratch_path=None, file_gb=2.0, mmap=mmap, gds=gds)
    monkeypatch.setattr(profiling, "nvme_to_hbm", lambda **_: result)
    assert ran_clean(run_cli("storage"))
    out = capsys.readouterr().out
    assert "mmap" in out and "gds" in out
    assert "2.00x the mmap bounce" in out


def test_storage_command_reports_skip_reason_without_speedup(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A run with no GDS read prints the mmap number alone and why GDS was skipped."""
    mmap = ReadResult(label="mmap", gigabytes_per_s=5.0, latency_ms=2.0)
    result = StorageBandwidth(
        available=True, scratch_path=None, file_gb=2.0, mmap=mmap, skipped="kvikio not installed"
    )
    monkeypatch.setattr(profiling, "nvme_to_hbm", lambda **_: result)
    assert ran_clean(run_cli("storage"))
    out = capsys.readouterr().out
    assert "mmap" in out
    assert "gds skipped: kvikio not installed" in out


def test_storage_command_prints_nothing_extra_with_no_speedup_or_skip_reason(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """With neither a speedup nor a skip reason, the command prints only the read lines."""
    mmap = ReadResult(label="mmap", gigabytes_per_s=5.0, latency_ms=2.0)
    result = StorageBandwidth(available=True, scratch_path=None, file_gb=2.0, mmap=mmap)
    monkeypatch.setattr(profiling, "nvme_to_hbm", lambda **_: result)
    assert ran_clean(run_cli("storage"))
    out = capsys.readouterr().out
    assert "mmap" in out
    assert "x the mmap bounce" not in out
    assert "gds skipped" not in out
