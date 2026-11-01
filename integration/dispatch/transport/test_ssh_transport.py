"""The ssh transport ends a link past its deadline with the whole tree it started."""

import signal
from pathlib import Path

import psutil
import pytest
from plumbum import local

from mainboard.dispatch import HostUnreachable, SshTransport
from mainboard.dispatch.transport.core import BoundedShellSession

from ...builders import survivors
from .builders import streaming


@pytest.mark.parametrize(
    "trap",
    [
        pytest.param("", id="a-tree-that-stops-when-asked"),
        pytest.param(
            "trap '' TERM; ", id="a-tree-that-ignores-the-request-and-is-killed-after-it"
        ),
    ],
)
def test_a_link_past_its_deadline_is_reported_unreachable_and_its_tree_ends(
    tmp_path: Path, trap: str
) -> None:
    worker = tmp_path / "worker"
    command = ("sh", "-c", f"{trap}sleep 60 & echo $! > {worker}; wait")

    with pytest.raises(HostUnreachable, match="ssh probe to 'gold' timed out after 1s"):
        SshTransport().invoke(command, "gold", operation="probe", timeout=1.0)

    assert survivors([int(worker.read_text(encoding="utf-8"))]) == []


@pytest.mark.parametrize(
    "ended",
    [
        pytest.param(signal.SIGINT, id="an-interrupt"),
        pytest.param(signal.SIGHUP, id="a-hangup"),
        pytest.param(signal.SIGTERM, id="a-terminate-request"),
    ],
)
def test_a_caller_ended_mid_stream_takes_the_link_it_started_with_it(
    tmp_path: Path, ended: signal.Signals
) -> None:
    with streaming(tmp_path) as (caller, link):
        caller.send_signal(ended)
        caller.wait(timeout=30)

        assert survivors([link]) == []


def test_closing_a_shell_session_ends_what_it_started() -> None:
    shell = local["sh"].popen()
    session = BoundedShellSession(shell, "utf-8", connect_timeout=5.0, host="gold")
    _, printed, _ = session.run("sleep 60 & echo $!")
    worker = psutil.Process(int(printed))

    session.close()

    assert (shell.wait(), survivors([worker])) == (-15, [])
