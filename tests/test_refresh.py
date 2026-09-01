import subprocess
from collections.abc import Sequence
from pathlib import Path

from mainboard._refresh import after_parent


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
