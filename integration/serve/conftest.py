import os
import signal
import subprocess
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import suppress

import pytest

from mainboard.serve.constants import Service


@pytest.fixture
def refusing() -> str:
    """A process-compose that answers `version` and refuses every YAML it is shown.

    It refuses as a strict one refuses a key it does not know.
    """
    return """#!/bin/sh
case "$1" in
  version) echo "Version:        v1.124.0" ;;
  up) echo "Error: yaml: unknown field 'bogus'" >&2; exit 1 ;;
esac
"""


@pytest.fixture
def stand_ins() -> list[Service]:
    """Stand-ins for the declared children, one of each kind: resident, a pass, one that is off."""
    return [
        Service(name="resident", command=("/bin/sleep", "600"), stop=3),
        Service(name="pass", command=("/bin/echo", "tick"), every="10s", timeout=8, stop=2),
        Service(name="sink", off="v2-events builds it"),
    ]


@pytest.fixture
def spawned() -> Iterator[Callable[..., subprocess.Popen[bytes]]]:
    """Start a process in a session of its own; every one still running is killed afterwards."""
    started: list[subprocess.Popen[bytes]] = []

    def spawn(argv: Sequence[str], env: dict[str, str] | None = None) -> subprocess.Popen[bytes]:
        process = subprocess.Popen(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            env=env,
        )
        started.append(process)
        return process

    yield spawn
    for process in started:
        with suppress(ProcessLookupError, PermissionError):
            os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


@pytest.fixture
def eventually() -> Callable[[Callable[[], bool], float], bool]:
    """Whether `holds()` comes true within `seconds`, looked at every 50 ms."""

    def within(holds: Callable[[], bool], seconds: float) -> bool:
        deadline = time.monotonic() + seconds
        while not holds():
            if time.monotonic() > deadline:
                return False
            time.sleep(0.05)
        return True

    return within
