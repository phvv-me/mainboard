"""What a terminal gets when `mb` logs to one."""

import subprocess
import sys

from .conftest import MB


def test_logging_to_a_terminal_starts_on_every_platform() -> None:
    """The console renderer starts on a terminal; it once crashed every command there."""
    probe = (
        "import sys\n"
        "class Terminal:\n"
        "    def isatty(self): return True\n"
        "    def write(self, text): sys.__stderr__.write(text)\n"
        "    def flush(self): pass\n"
        "sys.stderr = Terminal()\n"
        "from mb import logger\n"
        "logger.info('reached', ok=True)\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, timeout=60, check=False
    )
    assert done.returncode == 0, done.stderr
    assert "reached" in done.stderr
    assert MB is not None
