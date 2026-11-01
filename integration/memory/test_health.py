"""How memory is: the doctor reads the spool alone, `status` asks the upstream too."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

from mainboard.core import Verdict
from mainboard.memory.health import MemoryHealth
from mainboard.memory.keeps import Keep
from mainboard.memory.outbox import Outbox

from .fakes import Recorder


def test_the_doctor_row_passes_while_nothing_waits_and_never_asks_a_server(
    workspace: Path,
) -> None:
    row = MemoryHealth.of(Outbox(workspace)).section(datetime.now(UTC))
    assert (row.section, row.verdict, row.fix) == ("memory", Verdict.PASS, "")


def test_the_doctor_row_warns_with_the_count_and_age_of_what_waits(workspace: Path) -> None:
    outbox = Outbox(workspace)
    outbox.append(Keep.of("one"))
    outbox.append(Keep.of("two"))
    row = MemoryHealth.of(outbox).section(datetime.now(UTC) + timedelta(days=2))
    assert (row.verdict, row.fix) == (Verdict.WARN, "mb memory status")
    assert "2 keeps wait" in row.detail and "2.0 days" in row.detail


def test_the_doctor_row_answers_past_a_folder_named_as_a_spool_line(workspace: Path) -> None:
    outbox = Outbox(workspace)
    outbox.append(Keep.of("one"))
    outbox.spool.joinpath(f"{Keep.of('two').key}.jsonl").mkdir()
    row = MemoryHealth.of(outbox).section(datetime.now(UTC))
    assert (row.verdict, "1 keeps wait" in row.detail) == (Verdict.WARN, True)


def test_status_delivers_what_waits_then_reports_the_upstream(workspace: Path) -> None:
    outbox = Outbox(workspace)
    outbox.append(Keep.of("one"))
    health = MemoryHealth.of(outbox, Recorder())
    row = health.row()
    assert (health.waiting, row["login"], row["keeps"], row["caller"]) == (0, "yes", 1, "owner")


def test_status_while_the_upstream_is_down_keeps_the_spool_and_says_why(workspace: Path) -> None:
    outbox = Outbox(workspace)
    outbox.append(Keep.of("one"))
    health = MemoryHealth.of(outbox, Recorder(down=True))
    assert (health.waiting, health.remote, health.reason) == (1, None, "the recorder is down")
    assert health.row()["upstream"] == "http://recorder.invalid/mcp"
