import subprocess
import sys
from pathlib import Path
from time import sleep

import psutil
import pytest

from mainboard.runtime.tree import ProcessTree

# A command that starts a worker, says where it is, and then waits on it, the shape of a job
# whose dataloader or engine outlives the process that launched it.
_PARENT = """
import subprocess, sys, time
worker = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
open(sys.argv[1], "w").write(str(worker.pid))
time.sleep(60)
"""

# A command that will not stop when asked.
_STUBBORN = """
import signal, sys, time
signal.signal(signal.SIGTERM, signal.SIG_IGN)
open(sys.argv[1], "w").write("ready")
time.sleep(60)
"""


def started(code: str, tmp_path: Path) -> tuple[subprocess.Popen[bytes], Path]:
    """Start `code` and wait until it has written its marker file."""
    marker = tmp_path / "marker"
    process = subprocess.Popen([sys.executable, "-c", code, str(marker)])
    while not marker.exists() or not marker.read_text(encoding="utf-8"):
        assert process.poll() is None
        sleep(0.01)
    return process, marker


def test_ending_a_command_ends_every_worker_it_started(tmp_path: Path) -> None:
    """Signalling only the launcher would leave its worker holding the card."""
    process, marker = started(_PARENT, tmp_path)
    worker = psutil.Process(int(marker.read_text(encoding="utf-8")))
    tree = ProcessTree(process, grace=10.0)
    assert tree.wait(0.05) is None
    assert tree.stop() is False
    worker.wait(10)
    assert not worker.is_running()
    # Nothing is left to signal once the command is gone, and asking again is harmless.
    tree.terminate()
    tree.kill()


@pytest.mark.skipif(sys.platform == "win32", reason="Windows cannot ignore a termination")
def test_a_command_that_ignores_termination_is_killed_once_the_grace_runs_out(
    tmp_path: Path,
) -> None:
    process, _ = started(_STUBBORN, tmp_path)
    tree = ProcessTree(process, grace=0.2)
    assert tree.stop() is True
    assert tree.wait() == 137


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signal statuses")
def test_a_command_ended_by_a_signal_reports_what_a_shell_would() -> None:
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    process.terminate()
    assert ProcessTree(process).wait() == 143


def test_a_member_that_ends_while_the_tree_is_being_signalled_is_skipped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Gone:
        """A process that ended between being listed and being signalled."""

        def terminate(self) -> None:
            raise psutil.NoSuchProcess(0)

        def kill(self) -> None:
            raise psutil.NoSuchProcess(0)

    process = subprocess.Popen([sys.executable, "-c", "pass"])
    process.wait()
    tree = ProcessTree(process)
    assert tree.wait() == 0
    monkeypatch.setattr(tree, "_members", lambda: [Gone()])
    tree.terminate()
    tree.kill()
