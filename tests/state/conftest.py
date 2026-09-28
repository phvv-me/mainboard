import subprocess
import sys
from pathlib import Path

import pytest

from mainboard.state import Lake

# A separate process appending `count` events one commit each, attaching per append the way a
# CLI command would, tagged with its `writer` name so the rows can be told apart afterwards.
WRITER = """
import sys
from pathlib import Path
from mainboard.state import Lake

root, writer, count = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
lake = Lake.at(root)
for number in range(count):
    lake.append("events", [{"batch": writer, "job": str(number)}])
"""


@pytest.fixture
def lake(tmp_path: Path) -> Lake:
    """A freshly created lake in an empty workspace."""
    created = Lake.at(tmp_path)
    created.create()
    return created


def writers(root: Path, names: tuple[str, ...], count: int) -> list[subprocess.Popen[str]]:
    """One `WRITER` process per name, all started before any is waited on."""
    return [
        subprocess.Popen(
            [sys.executable, "-c", WRITER, str(root), name, str(count)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        for name in names
    ]


def finished(processes: list[subprocess.Popen[str]]) -> list[str]:
    """Wait for every process, answering the output of each that failed."""
    failures = []
    for process in processes:
        output, _ = process.communicate(timeout=300)
        if process.returncode:
            failures.append(output)
    return failures
