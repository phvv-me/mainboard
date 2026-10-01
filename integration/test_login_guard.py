"""The login-node guard stops a line that outgrows its share of the user's limit, and only that.

It runs where a login node would: a POSIX shell with `setsid`. The limit is set from what this
user already holds, so the test means the same on a busy workstation and on a bare runner.
"""

import getpass
import shutil
import subprocess

import psutil
import pytest

from mainboard.context import ExecutionPlan
from mainboard.dispatch.wrapping import _LOGIN_SHARE, guarded
from mainboard.manifest import HostProfile

_POSIX = pytest.mark.skipif(not shutil.which("setsid"), reason="needs a shell with setsid")

_HOG = "python3 -c 'import time; b = bytearray(1536 * 2**20); time.sleep(30)'; echo survived"


def _plan(limit_gb: float) -> ExecutionPlan:
    return ExecutionPlan(
        host="login", profile=HostProfile(login_memory_gb=limit_gb), env="default"
    )


def _above_held(headroom_gb: float) -> ExecutionPlan:
    """A plan whose login limit puts the guard `headroom_gb` above what this user holds now."""
    user = getpass.getuser()
    held = sum(
        process.info["memory_info"].rss
        for process in psutil.process_iter(["username", "memory_info"])
        if process.info["username"] and process.info["username"].endswith(user)
    )
    return _plan((held / 2**30 + headroom_gb) / _LOGIN_SHARE)


@_POSIX
def test_a_line_outgrowing_the_guard_is_stopped_whole() -> None:
    ran = subprocess.run(
        ["bash", "-c", guarded(_HOG, _above_held(0.5))], capture_output=True, text=True, timeout=60
    )
    assert ran.returncode == 137, ran.stderr
    assert "stopped" in ran.stderr
    assert "survived" not in ran.stdout


@_POSIX
def test_a_calm_line_keeps_its_own_status() -> None:
    ran = subprocess.run(
        ["bash", "-c", guarded("echo calm; exit 3", _above_held(0.5))],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert (ran.returncode, ran.stdout) == (3, "calm\n")


def test_no_limit_leaves_the_line_alone() -> None:
    assert guarded("true", _plan(0.0)) == "true"
