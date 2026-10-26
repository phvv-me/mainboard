"""What a terminal gets when `mb` logs to one."""

import subprocess
import sys

from ..constants import MB


def test_logging_to_a_terminal_starts_on_every_platform() -> None:
    """The console renderer starts on a terminal; it once crashed every command there."""
    probe = """import sys
class Terminal:
    def isatty(self): return True
    def write(self, text): sys.__stderr__.write(text)
    def flush(self): pass
sys.stderr = Terminal()
from mb import logger
logger.info('reached', ok=True)
"""
    done = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, timeout=60, check=False
    )
    assert done.returncode == 0, done.stderr
    assert "reached" in done.stderr
    assert MB is not None
