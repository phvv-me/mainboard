"""One lane on many hosts: collect its cells, group them, dispatch a job per group and host.

What every multi-card campaign repeated by hand. The lane's parametrization is the plan, a group
is one job running its cells as fresh processes (the runner's `--fresh`), and receipts come home
through the sweep `wait` and `monitor` already run.
"""

from __future__ import annotations

import re
import sys
from typing import TYPE_CHECKING, TextIO

from cyclopts import App
from patos import FrozenModel

from ..core.errors import MissionError

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    import pytest


class Cell(FrozenModel):
    """One collected cell of a lane, `file.py::test[key]`.

    key: the parametrize id, empty for an unparametrized lane.
    params: the parametrize values as text, by name.
    """

    nodeid: str
    key: str
    params: dict[str, str]


class Group(FrozenModel):
    """The cells (parametrize ids in collection order) one dispatched job runs.

    name: the shared parametrize value or the ordinal slice.
    """

    name: str
    ids: tuple[str, ...]


def grouped(cells: Sequence[Cell], *, by: str = "", per_job: int = 0) -> tuple[Group, ...]:
    """The cells as job groups: one per value of parametrize name `by`, else slices of
    `per_job`, else (`per_job` 0) all in one."""
    if by:
        named: dict[str, list[str]] = {}
        for cell in cells:
            if by not in cell.params:
                raise MissionError(f"{cell.nodeid} has no parametrize value named {by!r}")
            named.setdefault(cell.params[by], []).append(cell.key)
        return tuple(Group(name=name, ids=tuple(ids)) for name, ids in named.items())
    keys = [cell.key for cell in cells]
    if per_job <= 0:
        return (Group(name="all", ids=tuple(keys)),)
    return tuple(
        Group(name=f"slice{index // per_job}", ids=tuple(keys[index : index + per_job]))
        for index in range(0, len(keys), per_job)
    )


def node_of(target: str) -> str:
    """The node a target serves, the directory right under `experiments`, empty when none."""
    found = re.search(r"(?:^|/)experiments/([^/]+)/", target)
    return found.group(1) if found else ""


class Capture:
    """A pytest plugin that writes every collected item as one JSON line, then stops."""

    def __init__(self, out: TextIO = sys.stdout) -> None:
        self.out = out

    def pytest_collection_finish(self, session: pytest.Session) -> None:
        for item in session.items:
            lane, _, key = item.nodeid.partition("[")
            drawn = getattr(item, "callspec", None)
            params = {name: str(value) for name, value in getattr(drawn, "params", {}).items()}
            cell = Cell(nodeid=item.nodeid, key=key.removesuffix("]"), params=params)
            print("CELL " + cell.model_dump_json(), file=self.out, flush=True)


def parsed(text: str) -> tuple[Cell, ...]:
    """The cells a collection printed, one `CELL {json}` line each."""
    return tuple(
        Cell.model_validate_json(line.removeprefix("CELL "))
        for line in text.splitlines()
        if line.startswith("CELL ")
    )


app = App(name="lanes", help="Collect a lane's cells as JSON lines, inside the workspace env.")


@app.command
def collect(target: str) -> int:
    """Collect `target` (`path/to/file.py::test`) under pytest, one `CELL` line per cell."""
    import pytest as runner

    return int(
        runner.main(
            [target, "--collect-only", "-q", "-p", "no:randomly", "-p", "no:cacheprovider"],
            plugins=[Capture()],
        )
    )


def summary(hosts: Iterable[str], groups: Sequence[Group]) -> list[dict[str, str | int]]:
    """The dispatch plan as rows, one per host and group."""
    return [
        {"host": host, "group": group.name, "cells": len(group.ids), "ids": " ".join(group.ids)}
        for host in hosts
        for group in groups
    ]


if __name__ == "__main__":
    app()
