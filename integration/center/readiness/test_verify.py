"""The readiness suite's shell probe answers a status and both streams, or 127 for silence."""

import subprocess
import sys

import pytest

from mainboard.center.readiness.verify import _spawned

from ...fakes import failing

_PRINTS = (
    "import os, sys; print(os.environ['MB_PROBE']); print('err', file=sys.stderr); sys.exit(3)"
)


def test_a_probe_answers_its_status_and_both_streams_under_the_environment_it_was_given() -> None:
    assert _spawned([sys.executable, "-c", _PRINTS], {"MB_PROBE": "seen"}) == (3, "seen\nerr\n")


@pytest.mark.parametrize(
    "fault",
    [
        pytest.param(FileNotFoundError("absent"), id="a-program-that-cannot-start"),
        pytest.param(subprocess.TimeoutExpired("slow", 60.0), id="a-shell-past-its-timeout"),
    ],
)
def test_a_probe_that_cannot_answer_reads_as_a_missing_command(
    fault: Exception, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(subprocess, "run", failing(fault))

    assert _spawned(["slow"], {}) == (127, "")
