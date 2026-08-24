import json
from typing import TYPE_CHECKING

import pytest

from mainboard import Board, MissionError
from mainboard.batch import Topic
from mainboard.batch.runner import directory
from mainboard.cli import build

from .conftest import receipts

if TYPE_CHECKING:
    from pathlib import Path

    from ..conftest import Relayed

_SPEC = """
name = "smoke"

[[jobs]]
name = "gold-echo"
target = "gold"
command = "echo hi"
runtime_s = 30
"""


def written(root: Path) -> str:
    """The spec file a verb is pointed at, workspace-relative the way a caller types it."""
    (root / "smoke.toml").write_text(_SPEC)
    return "smoke.toml"


def test_preparing_prints_what_each_job_ships_and_what_the_batch_ships(
    depot: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The total rides in the table, so the one number a caller budgets from survives `--json`."""
    with pytest.raises(SystemExit, match="0"):
        build(depot)(["batch", "prepare", written(depot), "--json"])
    rows = json.loads(capsys.readouterr().out)
    assert [row["job"] for row in rows] == ["gold-echo", "total"]
    assert rows[0]["target"] == "gold"
    assert rows[1]["raw_bytes"] == rows[0]["raw_bytes"]


def test_pricing_prints_a_row_per_job_and_a_closing_total(
    depot: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit, match="0"):
        build(depot)(["batch", "estimate", written(depot), "--agent"])
    out = capsys.readouterr().out
    assert "setup_p50_s" in out
    assert "gold-echo\tgold\tssh" in out
    assert "\ntotal\t" in out


def test_a_batch_can_be_declared_without_a_file_at_all(
    depot: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit, match="0"):
        build(depot)(["batch", "prepare", "--job", "gold:echo hi", "--name", "quick", "--agent"])
    assert "gold-1\tgold" in capsys.readouterr().out


def test_a_batch_declared_as_nothing_is_refused_with_the_two_ways_to_declare_one(
    depot: Path,
) -> None:
    with pytest.raises(MissionError, match="a spec file, or --job"):
        build(depot)(["batch", "prepare"])


def test_running_prints_the_batch_id_then_every_handle_it_dispatched(
    depot: Path, relayed: list[Relayed], capsys: pytest.CaptureFixture[str]
) -> None:
    """The id is what `watch` is pointed at next, so a bare terminal run leads with it."""
    with pytest.raises(SystemExit, match="0"):
        build(depot)(["batch", "run", written(depot)])
    out = capsys.readouterr().out
    identity = out.splitlines()[0]
    assert identity.startswith("smoke-")
    assert "4242" in out
    assert [call[0] for call in relayed] == ["submit"]
    lines = receipts(directory(Board(depot), identity))
    assert [event.topic for event in lines.replay()] == [Topic.OPENED, Topic.SUBMITTED]


def test_a_compact_run_prints_the_record_alone_with_the_id_inside_it(
    depot: Path, relayed: list[Relayed], capsys: pytest.CaptureFixture[str]
) -> None:
    """A caller reading a record parses rows, so the bare id line would only be noise there."""
    with pytest.raises(SystemExit, match="0"):
        build(depot)(["batch", "run", written(depot), "--agent"])
    out = capsys.readouterr().out
    assert out.startswith("job\ttarget\thandle")
    assert "gold-echo\tgold\t4242\tpbs" in out


def test_watching_settles_every_target_in_one_table(
    depot: Path, relayed: list[Relayed], capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit, match="0"):
        build(depot)(["batch", "run", written(depot)])
    identity = capsys.readouterr().out.splitlines()[0]
    with pytest.raises(SystemExit, match="0"):
        build(depot)(["batch", "watch", identity])
    out = capsys.readouterr().out
    assert "0 running" in out
    assert "gold-echo" in out


def test_watching_follows_until_the_batch_closes(
    depot: Path, relayed: list[Relayed], capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit, match="0"):
        build(depot)(["batch", "run", written(depot)])
    identity = capsys.readouterr().out.splitlines()[0]
    with pytest.raises(SystemExit, match="0"):
        build(depot)(["batch", "watch", identity, "--interval", "0.01", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["job"] == "gold-echo"
