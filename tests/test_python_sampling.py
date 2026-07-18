import subprocess
from pathlib import Path

import pytest
from pydantic import ValidationError

from mainboard.profiling.python import (
    AsyncMode,
    PythonAction,
    PythonFormat,
    PythonMode,
    Tachyon,
)


def completed(
    command: tuple[str, ...],
    *,
    check: bool,
    capture_output: bool,
    text: bool = False,
    timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    """Return a deterministic subprocess result for command construction tests."""
    del check, capture_output, text, timeout
    return subprocess.CompletedProcess(command, 0, stdout="sample", stderr="warning")


def test_enum_auto_values_are_lowercase_names() -> None:
    assert PythonAction.RUN == "run"
    assert PythonMode.EXCEPTION == "exception"
    assert PythonFormat.FLAMEGRAPH == "flamegraph"
    assert AsyncMode.ALL == "all"


def test_run_builds_module_and_script_commands(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(subprocess, "run", completed)
    tachyon = Tachyon(executable=Path("python"))
    module = tachyon.run("pkg.mod", args=("--x",))
    script = tachyon.run("train.py", args=("1",))
    assert module.command[-3:] == ("pkg.mod", "--x") or module.command[-3:] == (
        "-m",
        "pkg.mod",
        "--x",
    )
    assert "-m" in module.command
    assert script.command[-2:] == ("train.py", "1")
    assert script.target == "train.py"


def test_every_sampling_option_is_forwarded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(subprocess, "run", completed)
    tachyon = Tachyon(
        executable=Path("python"),
        mode=PythonMode.EXCEPTION,
        format=PythonFormat.FLAMEGRAPH,
        sampling_rate="20khz",
        duration=2.5,
        all_threads=True,
        native=True,
        include_gc=False,
        opcodes=True,
        subprocesses=True,
        blocking=True,
        output=Path("profile.html"),
    )
    command = tachyon.run("x.py").command
    for option in (
        "--sampling-rate",
        "--mode=exception",
        "--duration",
        "--all-threads",
        "--native",
        "--no-gc",
        "--opcodes",
        "--subprocesses",
        "--blocking",
        "--flamegraph",
        "--output",
    ):
        assert option in command


def test_async_aware_options_are_forwarded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(subprocess, "run", completed)
    command = (
        Tachyon(
            executable=Path("python"),
            async_aware=True,
            async_mode=AsyncMode.ALL,
        )
        .run("x.py")
        .command
    )
    assert "--async-aware" in command
    assert "--async-mode=all" in command


@pytest.mark.parametrize(
    "options",
    (
        {"native": True},
        {"all_threads": True},
        {"mode": PythonMode.CPU},
        {"mode": PythonMode.GIL},
    ),
)
def test_invalid_combinations_raise(options: dict[str, bool | PythonMode]) -> None:
    with pytest.raises(ValidationError, match="async-aware"):
        Tachyon.model_validate({"async_aware": True, **options})
    with pytest.raises(ValidationError, match="opcode"):
        Tachyon(opcodes=True)
    with pytest.raises(ValidationError):
        Tachyon.model_validate({"duration": 0})


def test_attach_dump_and_replay(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(subprocess, "run", completed)
    tachyon = Tachyon(
        executable=Path("python"),
        format=PythonFormat.BINARY,
        output=Path("capture.bin"),
        all_threads=True,
        native=True,
        include_gc=False,
        blocking=True,
    )
    assert tachyon.attach(12).action is PythonAction.ATTACH
    dumped = tachyon.dump(12)
    assert dumped.action is PythonAction.DUMP
    assert dumped.mode is None and dumped.format is None
    replayed = tachyon.replay("capture.bin")
    assert replayed.action is PythonAction.REPLAY
    assert replayed.output == Path("capture.bin")

    async_dump = Tachyon(
        executable=Path("python"),
        format=PythonFormat.FLAMEGRAPH,
        async_aware=True,
        async_mode=AsyncMode.ALL,
        opcodes=True,
    ).dump(13)
    assert "--async-aware" in async_dump.command
    assert "--opcodes" in async_dump.command


def test_python_profile_string_prefers_text_then_artifact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(subprocess, "run", completed)
    result = Tachyon(output=Path("x.out")).run("x.py")
    assert str(result) == "sample"
    assert str(result.model_copy(update={"stdout": ""})) == "x.out"
    assert str(result.model_copy(update={"stdout": "", "output": None})) == ""


def test_available_handles_process_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(subprocess, "run", completed)
    assert Tachyon().available() is True

    def unavailable(
        command: tuple[str, ...],
        *,
        check: bool,
        capture_output: bool,
        timeout: float | None = None,
    ) -> subprocess.CompletedProcess[str]:
        del capture_output, timeout
        raise subprocess.CalledProcessError(1, command) if check else AssertionError

    monkeypatch.setattr(subprocess, "run", unavailable)
    assert Tachyon().available() is False
