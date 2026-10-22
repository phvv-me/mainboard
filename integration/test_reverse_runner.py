import subprocess
from pathlib import Path
from typing import BinaryIO

import pytest

from mainboard.core.tool import ToolRunner


@pytest.mark.parametrize("timeout", [False, True])
def test_runner_bounds_retained_output_and_preserves_timeout_evidence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    timeout: bool,
) -> None:
    def run(
        command: tuple[str, ...],
        *,
        cwd: Path | None,
        stdout: BinaryIO,
        stderr: BinaryIO,
        timeout: float,
        check: bool,
    ) -> subprocess.CompletedProcess[bytes]:
        assert command == ("fixture-tool", "literal;$(argument)")
        assert cwd == tmp_path and timeout == 1 and check is False
        stdout.write(b"123456789")
        stderr.write(b"error")
        if timed_out:
            raise subprocess.TimeoutExpired(command, timeout)
        return subprocess.CompletedProcess(command, 0)

    timed_out = timeout
    monkeypatch.setattr(subprocess, "run", run)
    found = ToolRunner(timeout=1, max_bytes=4).run(
        "fixture-tool",
        ("literal;$(argument)",),
        cwd=tmp_path,
    )
    assert found.stdout == "1234" and found.stderr == "erro"
    assert found.truncated and found.returncode == (124 if timeout else 0)


def test_runner_refuses_expired_deadline_before_starting_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("mainboard.core.tool.monotonic", lambda: 10.0)
    found = ToolRunner(deadline=9.0).run("must-not-execute", ())
    assert found.returncode == 124 and found.truncated
    assert found.stderr == "tool deadline exhausted"
