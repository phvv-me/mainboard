import threading
import time
from collections.abc import Callable
from pathlib import Path

from mainboard import Board
from mainboard.dispatch import Spool
from mainboard.dispatch.mirroring.spool import Submission
from mainboard.holding.serve import Server


def entry(spool: Spool, script: str, *, walltime: str = "00:05:00", name: str = "") -> Submission:
    """Publish a queued job running `script` the way the center would."""
    submission = Submission(
        handle=name or f"h{time.time_ns():x}",
        label="mainboard-test",
        script=script,
        cwd=str(spool.path),
        walltime=walltime,
        submitted="now",
    )
    assert spool.publish(
        spool.inbox, f"{submission.handle}.json", text=submission.model_dump_json()
    )
    return submission


def script(tmp_path: Path, body: str) -> str:
    """A job script file running `body`."""
    path = tmp_path / f"job-{time.time_ns():x}.sh"
    path.write_text(body, encoding="utf-8")
    return str(path)


def serve(spool: Spool, node: dict[str, str], *, walltime: str = "01:00:00") -> threading.Thread:
    """Run a server for `spool` in a thread, quickly polling, and answer the thread."""
    server = Server(
        spool, gen=1, walltime=walltime, environ=node, sleep=lambda _: time.sleep(0.02)
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    return thread


def halt(spool: Spool, thread: threading.Thread) -> None:
    """Ask the server in `thread` to stop and wait for it to end."""
    (spool.path / "stop").touch()
    thread.join(20)


def until(condition: Callable[[], bool], seconds: float = 20.0) -> None:
    """Wait for `condition()` to hold, failing the test after `seconds`."""
    deadline = time.monotonic() + seconds
    while not condition():
        assert time.monotonic() < deadline, "timed out waiting"
        time.sleep(0.02)


def pbs_board(center: Path, root: str) -> Board:
    """A workspace at `center` declaring the PBS host `miyabi-g`, whose workspace is `root`."""
    center.mkdir(exist_ok=True)
    manifest = f'[workspace]\nname = "it"\n[hosts.miyabi-g]\nkind = "pbs"\nroot = "{root}"\n'
    (center / "mb.toml").write_text(manifest, encoding="utf-8")
    return Board(center).on("miyabi-g")
