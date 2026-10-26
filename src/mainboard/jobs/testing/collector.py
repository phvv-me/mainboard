"""Workspace-side pytest collection; the ordinary CLI must not import this module."""

import sys
from typing import TextIO

import pytest
from cyclopts import App

from .lanes import Cell


class Capture:
    """Write collected cells to the parent process without running their tests."""

    def __init__(self, out: TextIO = sys.stdout) -> None:
        self.out = out

    def pytest_collection_finish(self, session: pytest.Session) -> None:
        for item in session.items:
            _, _, key = item.nodeid.partition("[")
            drawn = getattr(item, "callspec", None)
            params = {name: str(value) for name, value in getattr(drawn, "params", {}).items()}
            cell = Cell(nodeid=item.nodeid, key=key.removesuffix("]"), params=params)
            self.out.write(f"CELL {cell.model_dump_json()}\n")
            self.out.flush()


app = App(name="lanes", help="Collect a lane's cells as JSON lines, inside the workspace env.")


@app.command
def collect(target: str) -> int:
    """Collect `target` (`path/to/file.py::test`) under pytest, one `CELL` line per cell."""
    return int(
        pytest.main(
            [target, "--collect-only", "-q", "-p", "no:randomly", "-p", "no:cacheprovider"],
            plugins=[Capture()],
        )
    )


if __name__ == "__main__":
    app()
