import json
from typing import TYPE_CHECKING

import pytest

from mainboard import Board, ComputePath, Job, MissionError, Monitor, Survey
from mainboard.cli import build, main
from mainboard.compute import Access
from mainboard.dispatch import Handle
from mainboard.dispatch.state import DownHost, Failed, Finished, MonitorReport

if TYPE_CHECKING:
    from pathlib import Path

_FIELD_VALUE_HEADER = "field\tvalue"
_MIYABI_G = "miyabi-g"


def _fake_submit(self: Board, command: str, **options: str | int | None) -> Job:
    return Job(self, Handle(id="4242", host=self.host, root="/work/p", kind="pbs"))


def _swept() -> MonitorReport:
    """A report with one job of every outcome, so a render covers each row shape."""
    return MonitorReport(
        running=2,
        finished=[Finished(handle="1", target="gold", pulled_path="results/run")],
        failed=[Failed(handle="2", target="gold", reason="exited 137 (out of memory)")],
        unreachable_hosts=[DownHost(host=_MIYABI_G, reason="daemon down")],
    )


def _surveyed() -> list[ComputePath]:
    """One row of every shape a compute table can hold, so a render covers each cell."""
    return [
        ComputePath(name="local", kind="local", access=Access.HERE, detail="1x RTX 4090, 64 GB"),
        ComputePath(name=_MIYABI_G, kind="pbs", access=Access.UNREACHABLE, detail="timed out"),
        ComputePath(
            name="vast",
            kind="provider",
            access=Access.KEYED,
            detail="1x RTX 4090 Texas, US",
            usd_hr=0.31,
            credit_usd=42.5,
        ),
    ]


def test_plan_verb_prints_the_resolved_plan(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit, match="0"):
        build(workspace)(["plan", _MIYABI_G, "--json"])
    plan = json.loads(capsys.readouterr().out)
    assert plan["host"] == _MIYABI_G
    assert plan["container"]["image"].startswith("nvcr.io")


def test_plan_verb_default_is_a_rich_table(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit, match="0"):
        build(workspace)(["plan", _MIYABI_G])
    out = capsys.readouterr().out
    assert _MIYABI_G in out
    assert "plan" in out


def test_plan_verb_agent_mode_is_tabular(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit, match="0"):
        build(workspace)(["plan", _MIYABI_G, "--agent"])
    text = capsys.readouterr().out
    assert text.splitlines()[0] == _FIELD_VALUE_HEADER
    assert _MIYABI_G in text


def test_check_verb_lists_the_declared_surface(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit, match="0"):
        build(workspace)(["check", "--json"])
    surface = json.loads(capsys.readouterr().out)
    assert surface["workspace"] == "lab"
    assert _MIYABI_G in surface["hosts"]
    assert "ngc" in surface["containers"]


def test_check_verb_agent_mode_is_tabular(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit, match="0"):
        build(workspace)(["check", "--agent"])
    text = capsys.readouterr().out
    assert text.splitlines()[0] == _FIELD_VALUE_HEADER
    assert "workspace" in text
    assert "lab" in text


def test_check_verb_rejects_json_and_agent_together(workspace: Path) -> None:
    with pytest.raises(MissionError, match="only one"):
        build(workspace)(["check", "--json", "--agent"])


def test_check_verb_fields_flag_trims_and_drops_blank_entries(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit, match="0"):
        build(workspace)(["check", "--json", "--fields", "workspace, , hosts"])
    payload = json.loads(capsys.readouterr().out)
    assert set(payload) == {"workspace", "hosts"}


def test_root_discovery_walks_up_from_the_cwd(
    workspace: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    nested = workspace / "deep" / "inside"
    nested.mkdir(parents=True)
    monkeypatch.chdir(nested)
    with pytest.raises(SystemExit, match="0"):
        build()(["check", "--json"])
    assert json.loads(capsys.readouterr().out)["workspace"] == "lab"


def test_main_prints_mission_errors_without_traceback(
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(workspace)
    monkeypatch.setattr("sys.argv", ["mainboard", "plan", "gold", "--env", "ghost"])
    with pytest.raises(SystemExit, match="1"):
        main()
    assert "declared environments" in capsys.readouterr().err


def test_main_runs_a_clean_verb(
    workspace: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(workspace)
    monkeypatch.setattr("sys.argv", ["mainboard", "check"])
    with pytest.raises(SystemExit, match="0"):
        main()
    assert "lab" in capsys.readouterr().out


def test_run_verb_executes_locally(workspace: Path) -> None:
    with pytest.raises(SystemExit, match="0"):
        build(workspace)(["run", "true"])
    with pytest.raises(SystemExit, match="1"):
        build(workspace)(["run", "false"])


def test_run_verb_passes_leading_hyphen_tokens_through(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[str] = []
    monkeypatch.setattr(Board, "run", lambda self, command, **options: seen.append(command) or 0)
    with pytest.raises(SystemExit, match="0"):
        build(workspace)(["run", "python", "-c", "print(1)"])
    assert seen == ["python -c 'print(1)'"]


@pytest.mark.parametrize("flag", ["--version", "--help", "-h"])
def test_run_verb_hands_the_cli_s_own_flags_to_the_command_after_the_delimiter(
    workspace: Path, monkeypatch: pytest.MonkeyPatch, flag: str
) -> None:
    """`--version` and `--help` after `--` belong to the wrapped program, not to this CLI."""
    seen: list[str] = []
    monkeypatch.setattr(Board, "run", lambda self, command, **options: seen.append(command) or 0)
    with pytest.raises(SystemExit, match="0"):
        build(workspace)(["run", "--", "python", flag])
    assert seen == [f"python {flag}"]


@pytest.mark.parametrize("flag", ["--version", "--help", "-h"])
def test_submit_verb_hands_the_cli_s_own_flags_to_the_command_after_the_delimiter(
    workspace: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], flag: str
) -> None:
    monkeypatch.setattr(Board, "submit", _fake_submit)
    with pytest.raises(SystemExit, match="0"):
        build(workspace)(["submit", "--on", _MIYABI_G, "--", "python", flag])
    assert capsys.readouterr().out.strip() == "4242"


def test_run_verb_still_documents_itself_before_the_delimiter(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Dropping the version flag from the passthrough verbs must not cost them their help."""
    with pytest.raises(SystemExit, match="0"):
        build(workspace)(["run", "--help"])
    assert "container" in capsys.readouterr().out


def test_shell_verb_opens_the_named_environment_on_this_machine(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[tuple[str, str]] = []
    monkeypatch.setattr(Board, "shell", lambda self, env: seen.append((self.host, env)))
    with pytest.raises(SystemExit, match="0"):
        build(workspace)(["shell", "--env", "serving"])
    assert seen == [("local", "serving")]


def test_shell_verb_defaults_to_the_profiles_environment(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[tuple[str, str]] = []
    monkeypatch.setattr(Board, "shell", lambda self, env: seen.append((self.host, env)))
    with pytest.raises(SystemExit, match="0"):
        build(workspace)(["shell"])
    assert seen == [("local", "")]


def test_submit_verb_prints_the_handle(
    workspace: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(Board, "submit", _fake_submit)
    with pytest.raises(SystemExit, match="0"):
        build(workspace)(["submit", "--on", _MIYABI_G, "python", "-m", "exp.run"])
    assert capsys.readouterr().out.strip() == "4242"


def test_submit_verb_json_mode_prints_the_full_handle(
    workspace: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(Board, "submit", _fake_submit)
    with pytest.raises(SystemExit, match="0"):
        build(workspace)(["submit", "--on", _MIYABI_G, "--json", "python", "-m", "exp.run"])
    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "id": "4242",
        "host": _MIYABI_G,
        "root": "/work/p",
        "kind": "pbs",
        "fetch_path": None,
    }


def test_submit_verb_agent_mode_prints_the_tabular_handle(
    workspace: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(Board, "submit", _fake_submit)
    with pytest.raises(SystemExit, match="0"):
        build(workspace)(["submit", "--on", _MIYABI_G, "--agent", "python", "-m", "exp.run"])
    text = capsys.readouterr().out
    assert text.splitlines()[0] == _FIELD_VALUE_HEADER
    assert "4242" in text


def test_compute_verb_default_tables_every_path(
    workspace: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(workspace)
    monkeypatch.setattr(Survey, "paths", lambda self: _surveyed())
    with pytest.raises(SystemExit, match="0"):
        build(workspace)(["compute"])
    out = capsys.readouterr().out
    assert "local" in out and "vast" in out
    assert "unreachable" in out and "keyed" in out


def test_compute_verb_json_mode_prices_and_credits_the_provider_rows(
    workspace: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(workspace)
    monkeypatch.setattr(Survey, "paths", lambda self: _surveyed())
    with pytest.raises(SystemExit, match="0"):
        build(workspace)(["compute", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert [row["name"] for row in payload] == ["local", _MIYABI_G, "vast"]
    assert payload[0]["access"] == "here"
    assert payload[2] == {
        "name": "vast",
        "kind": "provider",
        "access": "keyed",
        "detail": "1x RTX 4090 Texas, US",
        "usd_hr": 0.31,
        "credit_usd": 42.5,
    }


def test_compute_verb_agent_mode_projects_the_named_columns(
    workspace: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(workspace)
    monkeypatch.setattr(Survey, "paths", lambda self: _surveyed())
    with pytest.raises(SystemExit, match="0"):
        build(workspace)(["compute", "--agent", "--fields", "name,credit_usd"])
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "name\tcredit_usd"
    assert lines[3] == "vast\t42.5"


def test_compute_verb_rejects_json_and_agent_before_probing_anything(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(self: Survey) -> list[ComputePath]:
        raise AssertionError("the mode flags are checked before any probe runs")

    monkeypatch.setattr(Survey, "paths", refuse)
    with pytest.raises(MissionError, match="only one"):
        build(workspace)(["compute", "--json", "--agent"])


def test_monitor_verb_json_mode_prints_the_whole_report(
    workspace: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(workspace)
    monkeypatch.setattr(Monitor, "once", lambda self: _swept())
    with pytest.raises(SystemExit, match="0"):
        build(workspace)(["monitor", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["running"] == 2
    assert payload["changed"] is True
    assert payload["finished"][0]["pulled_path"] == "results/run"
    assert payload["unreachable_hosts"][0]["host"] == _MIYABI_G


def test_monitor_verb_sweeps_an_untouched_cache_without_changes(
    workspace: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(workspace)
    with pytest.raises(SystemExit, match="0"):
        build(workspace)(["monitor", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "running": 0,
        "finished": [],
        "failed": [],
        "unreachable_hosts": [],
        "changed": False,
    }


def test_monitor_verb_default_tables_what_changed(
    workspace: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(workspace)
    monkeypatch.setattr(Monitor, "once", lambda self: _swept())
    with pytest.raises(SystemExit, match="0"):
        build(workspace)(["monitor"])
    out = capsys.readouterr().out
    assert "2 running" in out
    assert "unreachable" in out and "daemon down" in out
    assert "failed" in out


def test_monitor_verb_still_prints_its_heading_when_nothing_moved(
    workspace: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(workspace)
    with pytest.raises(SystemExit, match="0"):
        build(workspace)(["monitor"])
    out = capsys.readouterr().out
    assert "monitor: 0 running" in out
    assert "outcome" in out


def test_monitor_verb_agent_mode_is_tabular(
    workspace: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(workspace)
    monkeypatch.setattr(Monitor, "once", lambda self: _swept())
    with pytest.raises(SystemExit, match="0"):
        build(workspace)(["monitor", "--agent", "--fields", "running,changed"])
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == _FIELD_VALUE_HEADER
    assert [line.split("\t")[0] for line in lines[1:]] == ["running", "changed"]


def test_monitor_verb_watch_renders_every_pass(
    workspace: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(workspace)
    monkeypatch.setattr(Monitor, "watch", lambda self, interval: iter([_swept(), _swept()]))
    with pytest.raises(SystemExit, match="0"):
        build(workspace)(["monitor", "--watch", "0.1", "--json"])
    assert capsys.readouterr().out.count('"running": 2') == 2


def test_monitor_verb_watch_stops_quietly_on_an_interrupt(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def interrupted(self: Monitor, interval: float) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(Monitor, "watch", interrupted)
    with pytest.raises(SystemExit, match="0"):
        build(workspace)(["monitor", "--watch", "0.1"])
