"""An ssh link ends its process with everything that process started."""

import signal

from mainboard.dispatch.agent import SshLink

from ...builders import survivors


def test_ending_a_link_kills_its_process_and_everything_that_process_started(running) -> None:
    shell, members = running("sleep 60 & wait", workers=1)

    SshLink("gold").end(shell)

    assert (shell.returncode, survivors(members)) == (-signal.SIGKILL, [])
