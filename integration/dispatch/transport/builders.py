import os
import signal
import subprocess
import sys
import time
from collections.abc import Generator
from contextlib import contextmanager, suppress
from pathlib import Path

# A caller streaming through one link into a file, as a collection does. The link is `sh` writing
# its own pid and then sleeping, standing in for an ssh that has more to send.
_CALLER = """
import signal, sys
from pathlib import Path
from mainboard.dispatch import SshTransport

signal.signal(signal.SIGINT, signal.default_int_handler)
link = ("sh", "-c", f"echo $$ > {sys.argv[1]}; exec sleep 60")
SshTransport().run(link, "box", operation="stream", output=Path(sys.argv[2]))
"""


@contextmanager
def streaming(where: Path) -> Generator[tuple[subprocess.Popen[bytes], int]]:
    """A caller mid-stream and the pid of the link it started, the link killed on exit.

    where: the folder the caller's staging file and the link's pid file go in.
    """
    pidfile = where / "link"
    command = [sys.executable, "-c", _CALLER, str(pidfile), str(where / "streamed")]
    with subprocess.Popen(command) as caller:
        end = time.monotonic() + 30
        while not (pidfile.is_file() and pidfile.read_text(encoding="utf-8").strip()):
            assert caller.poll() is None and time.monotonic() < end, "the link never started"
            time.sleep(0.02)
        link = int(pidfile.read_text(encoding="utf-8"))
        try:
            yield caller, link
        finally:
            with suppress(ProcessLookupError):
                os.kill(link, signal.SIGKILL)
