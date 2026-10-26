"""One lane on many hosts: collect its cells, group them, dispatch a job per group and host.

What every multi-card campaign repeated by hand. The lane's parametrization is the plan, a group
is one job running its cells as fresh processes (the runner's `--fresh`), and receipts come home
through the sweep `wait` and `monitor` already run.
"""

import re
from typing import TYPE_CHECKING

from patos import FrozenModel

from ...core.errors import MissionError

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence


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
    """The cells as job groups.

    There is one group per value of parametrize name `by`, else slices of `per_job`, else
    (`per_job` 0) all in one.
    """
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


def parsed(text: str) -> tuple[Cell, ...]:
    """The cells a collection printed, one `CELL {json}` line each."""
    return tuple(
        Cell.model_validate_json(line.removeprefix("CELL "))
        for line in text.splitlines()
        if line.startswith("CELL ")
    )


def summary(hosts: Iterable[str], groups: Sequence[Group]) -> list[dict[str, str | int]]:
    """The dispatch plan as rows, one per host and group."""
    return [
        {"host": host, "group": group.name, "cells": len(group.ids), "ids": " ".join(group.ids)}
        for host in hosts
        for group in groups
    ]
