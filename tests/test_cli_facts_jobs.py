import json
from typing import TYPE_CHECKING

import pytest

from mainboard import Board
from mainboard.cli import build
from mainboard.dispatch import HostSetup
from mainboard.dispatch.state import Cache, RunRecord

if TYPE_CHECKING:
    from pathlib import Path

_FIELD_VALUE_HEADER = "field\tvalue"


def _seed_run(
    *,
    handle: str = "H1",
    target: str = "gold",
    name: str = "train",
    state: str = "ok",
    submitted_at: str = "2026-08-01T00:00:00",
) -> None:
    Cache().record(
        RunRecord(
            handle=handle,
            target=target,
            kind="ssh",
            script="job.sh",
            args="",
            git_sha="abc1234",
            dirty=0,
            submitted_at=submitted_at,
            name=name,
            state=state,
        )
    )


def test_facts_verb_prints_host_facts_json(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit, match="0"):
        build(workspace)(["facts", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] >= 1


def test_facts_verb_default_is_a_rich_table(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit, match="0"):
        build(workspace)(["facts"])
    out = capsys.readouterr().out
    assert "hostname" in out
    assert "facts" in out


def test_facts_verb_agent_mode_is_tabular(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit, match="0"):
        build(workspace)(["facts", "--agent"])
    text = capsys.readouterr().out
    assert text.splitlines()[0] == _FIELD_VALUE_HEADER
    assert "hostname" in text


def test_facts_verb_fields_projection_in_json_mode(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit, match="0"):
        build(workspace)(["facts", "--json", "--fields", "hostname,schema_version"])
    payload = json.loads(capsys.readouterr().out)
    assert set(payload) == {"hostname", "schema_version"}


def test_install_verb_delegates_to_the_board(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[tuple[str, str, bool, str]] = []

    def fake_install(self, env="default", *, resolve=False, profile="", watch=None):
        seen.append((self.host, env, resolve, profile))
        return HostSetup(host=self.host, root="/repo")

    monkeypatch.setattr(Board, "install", fake_install)
    with pytest.raises(SystemExit, match="0"):
        build(workspace)(["install", "serving", "--resolve", "--profile", "gold"])
    assert seen == [("local", "serving", True, "gold")]


def test_install_verb_targets_a_host_alias(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[str] = []

    def fake_install(self, env="default", *, resolve=False, profile="", watch=None):
        seen.append(self.host)
        return HostSetup(host=self.host, root="/repo")

    monkeypatch.setattr(Board, "install", fake_install)
    with pytest.raises(SystemExit, match="0"):
        build(workspace)(["install", "--on", "gold"])
    assert seen == ["gold"]


def test_jobs_verb_default_is_a_rich_table(
    workspace: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(workspace)
    _seed_run()
    with pytest.raises(SystemExit, match="0"):
        build(workspace)(["jobs"])
    out = capsys.readouterr().out
    assert "H1" in out
    assert "gold" in out
    assert "jobs" in out


def test_jobs_verb_json_mode_lists_the_projected_fields(
    workspace: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(workspace)
    _seed_run()
    with pytest.raises(SystemExit, match="0"):
        build(workspace)(["jobs", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload == [
        {
            "state": "ok",
            "host": "gold",
            "name": "train",
            "handle": "H1",
            "submitted_at": "2026-08-01T00:00:00",
        }
    ]


def test_jobs_verb_agent_mode_is_tabular(
    workspace: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(workspace)
    _seed_run()
    with pytest.raises(SystemExit, match="0"):
        build(workspace)(["jobs", "--agent"])
    text = capsys.readouterr().out.strip()
    lines = text.splitlines()
    assert lines[0] == "state\thost\tname\thandle\tsubmitted_at"
    assert "H1" in lines[1]


def test_jobs_verb_fields_projection(
    workspace: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(workspace)
    _seed_run()
    with pytest.raises(SystemExit, match="0"):
        build(workspace)(["jobs", "--json", "--fields", "handle,state"])
    payload = json.loads(capsys.readouterr().out)
    assert payload == [{"handle": "H1", "state": "ok"}]


def test_jobs_verb_with_no_recorded_runs_prints_nothing_alarming(
    workspace: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(workspace)
    with pytest.raises(SystemExit, match="0"):
        build(workspace)(["jobs"])
    assert "H1" not in capsys.readouterr().out


def test_jobs_verb_respects_the_limit(
    workspace: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(workspace)
    _seed_run(handle="H1", submitted_at="2026-08-01T00:00:00")
    _seed_run(handle="H2", submitted_at="2026-08-02T00:00:00")
    with pytest.raises(SystemExit, match="0"):
        build(workspace)(["jobs", "--json", "--limit", "1"])
    payload = json.loads(capsys.readouterr().out)
    assert [row["handle"] for row in payload] == ["H2"]


def _fake_setup(host: str = "gold") -> HostSetup:
    return HostSetup(
        host=host,
        root="/repo",
        env="default",
        activate="/repo/.mainboard/activate.sh",
        installer="uv",
        rejected=(("pip", "reported unavailable"),),
        tool="0.1.0",
        onboarded_at="2026-08-17T00:00:00+00:00",
    )


def test_setup_verb_onboards_the_host_and_prints_the_record(
    workspace: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    seen: list[tuple[str, str]] = []

    def fake_install(self, env="default", *, resolve=False, profile="", watch=None):
        seen.append((self.host, env))
        watch("probing")
        return _fake_setup(self.host)

    monkeypatch.setattr(Board, "install", fake_install)
    with pytest.raises(SystemExit, match="0"):
        build(workspace)(["setup", "gold", "--env", "serving", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert seen == [("gold", "serving")]
    assert payload["installer"] == "uv"
    assert payload["activate"].endswith(".mainboard/activate.sh")


def test_setup_verb_default_is_a_rich_table(
    workspace: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        Board,
        "install",
        lambda self, env="default", *, resolve=False, profile="", watch=None: _fake_setup(
            self.host
        ),
    )
    with pytest.raises(SystemExit, match="0"):
        build(workspace)(["setup", "gold"])
    out = capsys.readouterr().out
    assert "setup" in out
    assert "gold" in out


def test_hosts_verb_lists_the_recorded_onboardings(
    workspace: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(workspace)
    Cache().save_host(_fake_setup())
    with pytest.raises(SystemExit, match="0"):
        build(workspace)(["hosts", "--json"])
    [payload] = json.loads(capsys.readouterr().out)
    assert payload["host"] == "gold"
    assert payload["installer"] == "uv"
    assert payload["onboarded_at"]
