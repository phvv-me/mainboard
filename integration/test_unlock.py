"""The relay a detached ssh master asks its questions through.

A master started by `unlock` has no console, so its prompts reach the terminal through
`relay.sh` and a folder. The relay must work with whatever PATH the master hands it (from
PowerShell, one without Git's tools) and must give up once `unlock` is gone, or the master waits
forever. Both run the real script through the shell Git's ssh would run it with.
"""

import os
import subprocess
import time
from pathlib import Path

import pytest

from mainboard.core.host import WINDOWS
from mainboard.dispatch import keys


def _started(tmp_path: Path) -> tuple[subprocess.Popen[str], Path]:
    """The relay asking "Verification code:" with a PATH holding none of Git's tools."""
    script, relay = tmp_path / "relay.sh", tmp_path / "relay"
    script.write_text(keys.RELAY_SCRIPT, newline="\n")
    relay.mkdir()
    shell, bare = (
        (str(keys.client().with_name("sh.exe")), r"C:\Windows\System32")
        if WINDOWS
        else ("/bin/sh", "")
    )
    asking = subprocess.Popen(
        [shell, script.as_posix(), "Verification code:"],
        env={**os.environ, "PATH": bare, keys.RELAY: relay.as_posix()},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    for _ in range(100):
        if (relay / "prompt").exists():
            return asking, relay
        time.sleep(0.05)
    asking.kill()
    pytest.fail(f"the relay never left its prompt: {asking.communicate()[1]}")


def test_the_answer_comes_back_through_a_bare_path(tmp_path: Path) -> None:
    asking, relay = _started(tmp_path)
    assert (relay / "prompt").read_text(encoding="utf-8") == "Verification code:"
    (relay / "answer").write_text("123456\n", newline="\n")
    out, err = asking.communicate(timeout=10)
    assert (asking.returncode, out, err) == (0, "123456\n", "")
    assert not (relay / "answer").exists()


def test_the_relay_gives_up_when_unlock_is_gone(tmp_path: Path) -> None:
    asking, relay = _started(tmp_path)
    for path in relay.iterdir():
        path.unlink()
    relay.rmdir()
    asking.communicate(timeout=10)
    assert asking.returncode == 1
