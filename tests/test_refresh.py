import runpy
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import pytest

from mainboard._refresh import after_parent

# Past every pid Linux, macOS or Windows can hand out, so the launcher it names is already gone.
_VANISHED = 2**22 + 1


def test_the_worker_waits_for_its_parent_before_replacing_the_tool_and_records_the_result(
    tmp_path: Path,
) -> None:
    """The lock holder goes first, uv second, and its diagnostic remains after both exit."""
    events: list[str] = []
    log = tmp_path / "self-update.log"

    def wait(parent: int) -> None:
        events.append(f"wait:{parent}")

    def execute(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        events.append(f"run:{' '.join(command)}")
        return subprocess.CompletedProcess(command, 7, stdout="out\n", stderr="err\n")

    command = ("uv", "tool", "install", "mainboard")
    assert after_parent(314, command, log, wait=wait, execute=execute) == 7
    assert events == ["wait:314", "run:uv tool install mainboard"]
    assert log.read_text(encoding="utf-8") == "exit=7\nout\nerr\n"


def test_the_module_run_as_a_script_waits_on_a_vanished_parent_and_records_the_update(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The deferred worker is launched as `python -m`, so its own entry point is the contract."""
    log = tmp_path / "out" / "self-update.log"
    monkeypatch.setattr(
        sys,
        "argv",
        ["_refresh", str(_VANISHED), str(log), "--", sys.executable, "-c", "print(9)"],
    )
    with pytest.raises(SystemExit, match="0"):
        runpy.run_module("mainboard._refresh", run_name="__main__", alter_sys=True)
    assert log.read_text(encoding="utf-8") == "exit=0\n9\n"
