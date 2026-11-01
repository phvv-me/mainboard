"""`serve asks` lists to anyone; `serve answer` and `serve grant` are the owner's at a terminal."""

import os
import stat
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

from mainboard import MissionError
from mainboard.core.events import EventSpool
from mainboard.costs import Ledger
from mainboard.enums import Act
from mainboard.manifest import HostProfile
from mainboard.manifest.schema.services.policy import Policy
from mainboard.policy.asks import Ask, Asks
from mainboard.policy.grants import GrantsFile
from mainboard.policy.warden import Warden
from mainboard.state import Lake

_MARKERS = (
    "AI_AGENT",
    "CLAUDECODE",
    "CODEX_SANDBOX",
    "CODEX_CI",
    "CODEX_THREAD_ID",
    "OPENCODE",
    "GEMINI_CLI",
    "CURSOR_AGENT",
    "CLINE_ACTIVE",
)


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A home of the test's own, where the grants file goes."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    return home


@pytest.fixture
def person(home: Path) -> dict[str, str]:
    """The environment of the owner's own shell: no harness marker, no `MB_AGENT`."""
    left = {name: value for name, value in os.environ.items() if name not in _MARKERS}
    return {**left, "HOME": str(home), "MB_AGENT": "", "NO_COLOR": "1"}


def _asked(workspace: Path, target: str) -> str:
    """The id of the ask a refused push of `target` filed."""
    with pytest.raises(MissionError) as refused:
        Warden(workspace, Policy()).push(target)
    return str(refused.value).split("asked the owner as ")[1][:8]


def test_serve_asks_lists_a_waiting_ask_to_an_agent(mb, workspace: Path, home: Path) -> None:
    short = _asked(workspace, "me/repo:main")
    ran = mb("serve", "asks", env={"MB_AGENT": "1", "HOME": str(home)})
    assert ran.code == 0
    assert f"{short}\tpush\tme/repo:main\t" in ran.out


def test_serve_asks_prints_no_escape_to_an_agent_when_a_forged_ask_waits(
    mb, workspace: Path, home: Path
) -> None:
    short = _asked(workspace, "me/repo:main")
    why = "ok\x1b[2J\x1b]52;c;cHduZWQ=\x07"
    forged = Ask.model_construct(act=Act.PUSH, target="me/x:y", why=why, at=datetime.now(UTC))
    with EventSpool(Asks(workspace).lake.spool / "policy-0.jsonl") as spool:
        spool.append(forged.event())
    ran = mb("serve", "asks", env={"MB_AGENT": "1", "HOME": str(home)})
    assert (ran.code, "\x1b" in ran.out + ran.err, "me/x:y" in ran.out) == (0, False, False)
    assert f"{short}\tpush\tme/repo:main\t" in ran.out


@pytest.mark.parametrize(
    "words",
    [
        pytest.param(("serve", "answer", "abcd1234"), id="an-answer"),
        pytest.param(("serve", "answer", "abcd1234", "--decline"), id="a-no"),
        pytest.param(("serve", "grant", "push", "me/repo:mb/*"), id="a-grant"),
        pytest.param(("serve", "grant", "push", "me/repo:mb/*", "--revoke"), id="a-revocation"),
        pytest.param(("serve", "grant", "card", "lambda", "--usd", "100"), id="a-card-limit"),
    ],
)
@pytest.mark.parametrize(
    "declared",
    [
        pytest.param({"MB_AGENT": "1"}, id="an-agent-audience"),
        pytest.param({"CLAUDECODE": "1", "MB_AGENT": "0"}, id="a-harness-saying-it-is-a-person"),
        pytest.param({"CODEX_THREAD_ID": "t", "MB_AGENT": "0"}, id="codex-saying-it-is-a-person"),
    ],
)
def test_an_agent_answering_or_granting_is_refused_and_no_grant_is_written(
    mb, workspace: Path, home: Path, words: tuple[str, ...], declared: dict[str, str]
) -> None:
    ran = mb(*words, env={**declared, "HOME": str(home)})
    assert (ran.code, ran.out) == (1, "")
    assert "is the owner's act" in ran.err
    assert not GrantsFile(workspace).path.parent.exists()


def test_the_owner_granting_at_a_terminal_lets_the_pushes_its_scope_covers_through(
    at_a_terminal, workspace: Path, person: dict[str, str]
) -> None:
    grant = [sys.executable, "-m", "mainboard", "serve", "grant", "push", "me/repo:mb/*"]
    ran = at_a_terminal([*grant, "--hours", "1"], env=person)
    Warden(workspace, Policy()).push("me/repo:mb/save")
    mode = stat.S_IMODE(GrantsFile(workspace).path.stat().st_mode)
    assert (ran.code, mode) == (0, 0o600), ran.said


def test_the_owner_declaring_a_card_limit_at_a_terminal_lets_that_card_cap_a_rental(
    at_a_terminal, workspace: Path, person: dict[str, str]
) -> None:
    declare = [sys.executable, "-m", "mainboard", "serve", "grant", "card", "lambda", "--usd"]
    refused = at_a_terminal([*declare, "100", "--revoke"], env=person)
    ran = at_a_terminal([*declare, "100"], env=person)
    ledger = Ledger(Lake.at(workspace).session())
    cap = Warden(workspace, Policy()).rent(HostProfile(kind="lambda"), max_usd=5.0, ledger=ledger)
    assert (refused.code, ran.code, cap.limit_usd) == (1, 0, 100.0), ran.said


def test_the_owner_answering_an_ask_grants_exactly_its_act_and_target(
    at_a_terminal, workspace: Path, person: dict[str, str]
) -> None:
    short = _asked(workspace, "me/repo:main")
    ran = at_a_terminal([sys.executable, "-m", "mainboard", "serve", "answer", short], env=person)
    Warden(workspace, Policy()).push("me/repo:main")
    assert ran.code == 0, ran.said
    with pytest.raises(MissionError, match="no live grant covers push me/repo:main2"):
        Warden(workspace, Policy()).push("me/repo:main2")


@pytest.mark.parametrize("hours", ["inf", "nan", "1e12", "-1", "0", "1e-300"])
@pytest.mark.parametrize("verb", ["grant", "answer"])
def test_the_owner_giving_hours_no_grant_can_last_is_refused_and_no_grant_is_written(
    at_a_terminal, workspace: Path, person: dict[str, str], verb: str, hours: str
) -> None:
    words = (
        ["grant", "push", "me/repo:mb/*"]
        if verb == "grant"
        else ["answer", _asked(workspace, "me/repo:main")]
    )
    ran = at_a_terminal(
        [sys.executable, "-m", "mainboard", "serve", *words, f"--hours={hours}"], env=person
    )
    assert (ran.code, "Traceback" in ran.said, "give --hours above 0" in ran.said) == (
        1,
        False,
        True,
    )
    assert not GrantsFile(workspace).path.exists()
