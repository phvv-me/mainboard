import os
import socket
import time
from pathlib import Path

import pytest

from mainboard.dispatch import Spool
from mainboard.dispatch.mirroring.spool import LineSpec, spool_path
from mainboard.dispatch.schedulers import Held, Pbs


@pytest.fixture
def root(tmp_path: Path) -> str:
    """A host workspace root; the spool and the runner's logs live beneath it."""
    return str(tmp_path)


@pytest.fixture
def spool(root: str) -> Spool:
    """The node's view of the spool at `root`, with a line open for an hour."""
    opened = Spool(Path(spool_path(root)))
    line = LineSpec(
        queue="interact-g",
        walltime="02:00:00",
        mem_gb=100,
        account="g",
        deadline=time.time() + 3600,
        created=time.time(),
        tool="mainboard",
    )
    opened.replace(opened.path / "line.json", line.model_dump_json())
    return opened


@pytest.fixture
def node(tmp_path: Path) -> dict[str, str]:
    """The two variables that make a process look like it runs inside a PBS allocation."""
    nodefile = tmp_path / "nodefile"
    nodefile.write_text(socket.gethostname() + "\n", encoding="utf-8")
    return {**os.environ, "PBS_JOBID": "123.opbs", "PBS_NODEFILE": str(nodefile)}


@pytest.fixture
def held(monkeypatch: pytest.MonkeyPatch) -> Held:
    """The scheduler with PBS itself absent: no allocation is ever reported ended."""
    monkeypatch.setattr(Pbs, "states", lambda self, remote, root, handles: {})
    return Held()
