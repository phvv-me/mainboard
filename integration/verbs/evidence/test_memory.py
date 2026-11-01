"""`memory find`, `keep` and `status`: rows for an agent, a keep that queues, a named extra.

The upstream is a fake in these calls; the subprocess calls run where AIZK's client is absent,
which is the refusal a missing extra gets.
"""

import importlib.util
import json
from pathlib import Path

import pytest

from mainboard.memory.keeps import Keep
from mainboard.memory.outbox import Outbox
from mainboard.render import Output
from mainboard.verbs.evidence.memory import MemoryVerbs
from mainboard.verbs.session import Session

from ...memory.fakes import Recorder

_AGENT = {"MB_AGENT": "1"}
_JSON = Output(json_=True)
_ABSENT = pytest.mark.skipif(
    importlib.util.find_spec("aizk_client") is not None,
    reason="the memory extra is installed here, so nothing refuses",
)


def _verbs(workspace: Path, upstream: Recorder) -> MemoryVerbs:
    return MemoryVerbs(session=Session(root=workspace), upstream=upstream)


def test_keep_prints_a_row_that_says_queued_while_the_upstream_is_down(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    upstream = Recorder(down=True)
    _verbs(workspace, upstream).keep("a note", output=_JSON)
    (row,) = json.loads(capsys.readouterr().out)
    assert (row["state"], row["item"], len(Outbox(workspace).waiting())) == ("queued", "a note", 1)


def test_keep_delivers_what_waits_and_says_kept_once_the_upstream_answers(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    upstream = Recorder()
    _verbs(workspace, upstream).keep("one", "two", note="both", scope="personal", output=_JSON)
    rows = json.loads(capsys.readouterr().out)
    assert ([row["state"] for row in rows], [row["document"] for row in rows]) == (
        ["kept", "kept"],
        ["document-1", "document-2"],
    )


def test_find_prints_the_answer_whole_after_the_spool_is_delivered(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    upstream = Recorder(down=True)
    verbs = _verbs(workspace, upstream)
    verbs.keep("remember gold", output=_JSON)
    capsys.readouterr()
    upstream.down = False
    verbs.find("what is gold")
    assert (capsys.readouterr().out, upstream.calls[0]) == (
        "answer to what is gold\n",
        ("keep", "remember gold"),
    )


@pytest.mark.parametrize("agent", ["1", "0"], ids=["an-agent", "a-person"])
def test_find_prints_an_answer_that_holds_terminal_controls_without_one_escape(
    workspace: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    agent: str,
) -> None:
    monkeypatch.setenv("MB_AGENT", agent)
    _verbs(workspace, Recorder()).find("\x1b[2J\x1b]52;c;aGVsbG8=\x07 what is gold")
    said = capsys.readouterr()
    assert ("\x1b" in said.out + said.err, "what is gold" in said.out) == (False, True)


@pytest.mark.parametrize(
    "place",
    [pytest.param(".env", id="the-workspace-dotenv"), pytest.param("~/.ssh/id_rsa", id="a-key")],
)
def test_keep_of_a_file_that_may_hold_secrets_exits_1_and_says_and_spools_none_of_it(
    mb, workspace: Path, place: str
) -> None:
    home = workspace / "home"
    for secret in (workspace / ".env", home / ".ssh" / "id_rsa"):
        secret.parent.mkdir(parents=True, exist_ok=True)
        secret.write_text("OPENAI_API_KEY=sk-canary-0123456789abcdef\n", encoding="utf-8")
    ran = mb("memory", "keep", place, env={**_AGENT, "HOME": str(home)})
    assert (ran.code, "canary" in ran.said, Outbox(workspace).waiting()) == (1, False, [])
    assert "may hold secrets" in ran.said


def test_status_answers_past_a_folder_named_as_a_spool_line(mb, workspace: Path) -> None:
    outbox = Outbox(workspace)
    outbox.append(Keep.of("x"))
    outbox.spool.joinpath(f"{Keep.of('y').key}.jsonl").mkdir()
    home = str(workspace / "home")
    ran = mb("memory", "status", env={**_AGENT, "HOME": home, "XDG_CONFIG_HOME": home})
    fields = dict(line.split("\t", 1) for line in ran.out.splitlines())
    assert (ran.code, fields["queued"]) == (0, "1")


def test_status_prints_the_record_of_an_upstream_that_answers(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _verbs(workspace, Recorder()).status(output=_JSON)
    record = json.loads(capsys.readouterr().out)
    assert (record["login"], record["queued"], record["keeps"]) == ("yes", 0, 0)


@_ABSENT
def test_status_for_an_agent_prints_plain_rows_with_the_extra_named_as_the_reason(
    mb, workspace: Path
) -> None:
    Outbox(workspace).append(Keep.of("x"))
    ran = mb("memory", "status", env=_AGENT)
    fields = dict(line.split("\t", 1) for line in ran.out.splitlines())
    assert (ran.code, fields["queued"], "\x1b" in ran.out + ran.err) == (0, "1", False)
    assert "mainboard[memory]" in fields["unanswered"]


@_ABSENT
@pytest.mark.parametrize(
    "words",
    [
        pytest.param(("keep", "a note"), id="keep"),
        pytest.param(("find", "what is gold"), id="find"),
    ],
)
def test_a_verb_that_needs_aizk_is_refused_by_naming_the_extra_and_spools_nothing(
    mb, workspace: Path, words: tuple[str, ...]
) -> None:
    ran = mb("memory", *words, env=_AGENT)
    assert (ran.code, "mainboard[memory]" in ran.err, Outbox(workspace).waiting()) == (1, True, [])


def test_keep_with_nothing_to_keep_is_refused(mb) -> None:
    ran = mb("memory", "keep", env=_AGENT)
    assert (ran.code, "name what to keep" in ran.err) == (1, True)


def test_the_drain_pass_with_no_keep_waiting_delivers_none_and_asks_no_server(
    mb, workspace: Path
) -> None:
    ran = mb("serve", "process", "drain", env=_AGENT)
    assert (ran.code, "drained 0 queued keeps into memory" in ran.err) == (0, True)
