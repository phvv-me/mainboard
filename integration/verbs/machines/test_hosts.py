"""What `host list` adds: each provider's cap on its bill, and each target's landings."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from mainboard import Board
from mainboard.dispatch.state import RunRecord
from mainboard.verbs.machines.hosts import Detail, HostVerbs

from .readers import table


def test_host_list_shows_the_cap_on_the_bill_of_each_provider(
    bare: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    HostVerbs().compute()
    rows = {row["name"]: row["cap"] for row in table(capsys.readouterr().out)}
    assert (rows["local"], rows["shadeform"], rows["lambda"]) == (
        "",
        "threshold",
        "card, no limit declared",
    )


def test_host_list_reliability_adds_the_landings_the_lake_kept(
    measured: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ran = RunRecord(
        handle="vast-1",
        target="vast",
        kind="vast",
        script="job.sh",
        args="",
        submitted_at=datetime.now(UTC).isoformat(),
        verdict="ok",
        exit_code=0,
    )
    Board(measured).dispatcher.cache.record(ran)
    HostVerbs().compute(detail=Detail(reliability=True))
    (row,) = table(capsys.readouterr().out)
    assert (row["target"], row["landings"], row["landed"], row["ready_p50_s"]) == (
        "vast",
        "5",
        "3",
        "120.0",
    )
