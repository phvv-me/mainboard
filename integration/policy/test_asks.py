"""An ask waits, spooled once per act and target, until the owner answers or a grant covers it."""

import json
from datetime import timedelta
from pathlib import Path

import pytest

from mainboard import MissionError
from mainboard.constants import ASK_EVENT
from mainboard.core.events import EventSpool
from mainboard.enums import Act
from mainboard.manifest.schema.services.policy import Policy
from mainboard.policy.asks import Ask, Asks
from mainboard.policy.grants import Declined, Grants
from mainboard.policy.warden import Warden
from mainboard.state.lake import Access

from ..state.telemetry.builders import lake_with
from .builders import ask, grant

_ASKED = ask()
_LATER = _ASKED.at + timedelta(minutes=30)
_BEFORE = _ASKED.at - timedelta(hours=3)
_LAKE_ROW = (
    "INSERT INTO lake.main.otlp_logs (time_unix_nano, service_name, event_name, body, "
    "log_attributes) VALUES (?, 'mb', ?, ?, ?)"
)


def test_asking_the_same_act_on_the_same_target_twice_spools_one_line(workspace: Path) -> None:
    asks = Asks(workspace)
    first = asks.file(_ASKED, Grants())
    again = asks.file(ask(at=_LATER), Grants())
    lines = [line for path in asks.lake.spool.glob("*.jsonl") for line in path.open()]
    assert (again, asks.pending(Grants(), at=_LATER), len(lines)) == (first, [first], 1)


@pytest.mark.parametrize(
    ("grants", "waiting"),
    [
        pytest.param(Grants(), True, id="no-answer-yet"),
        pytest.param(
            Grants(declined=(Declined(ask=_ASKED.id, at=_LATER),)), False, id="declined-since"
        ),
        pytest.param(
            Grants(declined=(Declined(ask=_ASKED.id, at=_BEFORE),)),
            True,
            id="asked-anew-after-a-no",
        ),
        pytest.param(
            Grants(grants=(grant("me/repo:main", granted=_LATER, ask=_ASKED.id),)),
            False,
            id="granted-since",
        ),
        pytest.param(
            Grants(
                grants=(
                    grant(
                        "me/repo:main",
                        granted=_BEFORE,
                        until=_BEFORE + timedelta(hours=1),
                        ask=_ASKED.id,
                    ),
                )
            ),
            True,
            id="asked-anew-after-its-grant-ended",
        ),
        pytest.param(Grants(grants=(grant("me/repo:*"),)), False, id="covered-by-a-grant-unasked"),
        pytest.param(
            Grants(grants=(grant("me/repo:main", act=Act.LAUNCH),)),
            True,
            id="a-grant-of-another-act-on-its-target",
        ),
    ],
)
def test_an_ask_waits_until_answered_since_or_covered_by_a_live_grant(
    workspace: Path, grants: Grants, waiting: bool
) -> None:
    asks = Asks(workspace)
    asks.file(_ASKED, Grants())
    assert (asks.pending(grants, at=_LATER) == [_ASKED]) == waiting


def test_asks_delivered_to_the_lake_are_read_beside_the_spool_and_a_forged_one_skipped(
    workspace: Path,
) -> None:
    delivered = ask("me/repo:mb/save", at=_BEFORE)
    forged = Ask.model_construct(act=Act.PUSH, target="me/repo:mb/../main", why="", at=_BEFORE)
    rows = [
        (row.at.replace(tzinfo=None), ASK_EVENT, row.why, json.dumps(row.event().attributes))
        for row in (delivered, forged)
    ]
    with lake_with(workspace, []).attachment.attached(access=Access.WRITE) as connection:
        connection.executemany(_LAKE_ROW, rows)
    asks = Asks(workspace)
    asks.file(_ASKED, Grants())
    assert asks.pending(Grants(), at=_LATER) == [delivered, _ASKED]


def test_a_spool_file_holding_a_line_that_is_no_event_is_skipped_and_the_rest_still_read(
    workspace: Path,
) -> None:
    asks = Asks(workspace)
    asks.file(_ASKED, Grants())
    (asks.lake.spool / "policy-0.jsonl").write_text("not an event\n", encoding="utf-8")
    with pytest.raises(MissionError, match="no live grant covers push me/repo:mb/save; asked"):
        Warden(workspace, Policy()).push("me/repo:mb/save")
    waiting = {known.target for known in asks.pending(Grants(), at=_LATER)}
    assert waiting == {"me/repo:main", "me/repo:mb/save"}


def test_a_spool_file_that_is_no_spool_is_said_once_however_often_it_is_read(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    asks = Asks(workspace)
    asks.file(_ASKED, Grants())
    (asks.lake.spool / "policy-0.jsonl").write_text("not an event\n", encoding="utf-8")
    for _ in range(2):
        with pytest.raises(MissionError):
            Warden(workspace, Policy()).push("me/repo:mb/save")
        asks.pending(Grants(), at=_LATER)
    assert capsys.readouterr().err.count("skipped policy-0.jsonl") == 1


@pytest.mark.parametrize(
    "why",
    [
        pytest.param("ok\x1b[2J", id="a-clear-screen"),
        pytest.param("ok\x1b]52;c;cHduZWQ=\x07", id="an-osc-52-clipboard-write"),
        pytest.param("ok\x9b2J", id="a-c1-control-sequence-introducer"),
        pytest.param("ok‮done", id="a-right-to-left-override"),
    ],
)
def test_an_ask_line_whose_why_holds_a_control_is_skipped_and_the_rest_still_read(
    workspace: Path, why: str
) -> None:
    asks = Asks(workspace)
    asks.file(_ASKED, Grants())
    forged = Ask.model_construct(act=Act.PUSH, target="me/repo:mb/save", why=why, at=_ASKED.at)
    with EventSpool(asks.lake.spool / "policy-0.jsonl") as spool:
        spool.append(forged.event())
    assert asks.pending(Grants(), at=_LATER) == [_ASKED]
