import json
import os
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from shutil import rmtree
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

from mainboard import Board, ComputePath, MissionError, Project, Survey
from mainboard.batch import JobEstimate
from mainboard.cli import build, main
from mainboard.dispatch.shared import db_file
from mainboard.dispatch.state import Cache, MonitorReport, RunRecord
from mainboard.dispatch.vocabulary import JobState
from mainboard.jobs.lanes import Cell
from mainboard.monitor import Monitor
from mainboard.probe.occupancy import CardOccupancy, Holder, Occupancy
from mainboard.verdicts import StreamVerdict, Verdicts

if TYPE_CHECKING:
    from pydantic import JsonValue

    from .support import Launcher, Relayed

_FIELD_VALUE_HEADER = "field\tvalue"
_MIYABI_G = "miyabi-g"

# What `submit` translates its flags into, every resource the verb carries, so a case naming one
# of them says what it changed and nothing else has to be repeated.
_RESOURCES = {
    "name": "",
    "queue": "",
    "walltime": "",
    "mem_gb": 0,
    "gpus": 0,
    "gpu_name": "",
    "max_usd": 0.0,
    "attempt": 1,
    "fetch": None,
    "node": "",
    "needs": (),
    "env": "",
    "container": "",
}


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (
            ["run", "--on", "gold", "--env", "serving", "--", "python", "-c", "print(1)"],
            (
                "run",
                "gold",
                (("python", "-c", "print(1)"),),
                {"env": "serving", "container": ""},
            ),
        ),
        (
            ["submit", "--on", _MIYABI_G, "--queue", "short-g", "--mem-gb", "64", "true"],
            ("submit", _MIYABI_G, ("true",), {**_RESOURCES, "queue": "short-g", "mem_gb": 64}),
        ),
        (
            ["add", "tqdm", "-l", "python", "--dev", "--no-resolve"],
            (
                "add",
                "",
                ("tqdm",),
                {"ecosystem": "python", "env": "", "dev": True, "resolve": False},
            ),
        ),
        (
            ["remove", "tqdm"],
            ("remove", "", ("tqdm",), {"ecosystem": "", "env": "", "dev": False, "resolve": True}),
        ),
        (
            ["upgrade", "--env", "serving"],
            ("upgrade", "", ("",), {"ecosystem": "", "env": "serving", "dev": False}),
        ),
        (
            ["new", "p", "--answer", "home=standalone", "--answer", "paper=draft"],
            (
                "render",
                "",
                ("p",),
                {
                    "template": "",
                    "description": "",
                    "dest": "",
                    "answers": {"home": "standalone", "paper": "draft"},
                },
            ),
        ),
        (["doctor"], ("sections", "", (), {})),
        (
            ["install", "serving", "--resolve", "--profile", "gold"],
            ("install", "local", ("serving",), {"resolve": True, "profile": "gold"}),
        ),
        (
            ["install", "--on", "gold"],
            ("install", "gold", ("",), {"resolve": False, "profile": ""}),
        ),
        (
            ["setup", "gold", "--env", "serving"],
            ("install", "gold", ("serving",), {"resolve": False, "sync_only": False}),
        ),
        (
            ["setup", "gold", "--sync-only"],
            ("install", "gold", ("",), {"resolve": False, "sync_only": True}),
        ),
        (
            ["sync", "gold", "--env", "serving"],
            ("install", "gold", ("serving",), {"resolve": False, "sync_only": True}),
        ),
        (["shell", "--env", "serving"], ("shell", "local", ("serving",), {})),
        (["serve", "vserve", "--on", "gold"], ("serve", "gold", ("vserve",), {})),
        (
            ["interact", "--on", "gold", "--queue", "interact-g", "--", "pwd"],
            (
                "interact",
                "gold",
                ("pwd",),
                {"env": "", "queue": "interact-g", "walltime": "", "keep": False},
            ),
        ),
        (["compute"], ("paths", "", (), {})),
        (["monitor"], ("once", "", (), {})),
        (["facts", "gold"], ("facts", "gold", (), {})),
        (
            ["wait", "4242", "--timeout", "60"],
            ("wait", "", ("4242",), {"host": "", "timeout": 60.0, "interval": 5.0}),
        ),
        (
            ["batch", "wait", "smoke-1", "--timeout", "60"],
            ("wait", "", ("smoke-1",), {"host": "", "timeout": 60.0, "interval": 5.0}),
        ),
        (["verdict", "smoke-1"], ("of", "", ("smoke-1",), {"host": "", "run": ""})),
        (["cancel", "4242", "--on", "gold"], ("cancel", "", ("4242",), {"host": "gold"})),
        (["logs", "4242"], ("captured", "", ("4242",), {"host": ""})),
        (["attest", "smoke-1"], ("attest", "local", ("smoke-1",), {"job": "smoke-1"})),
        (
            ["attest", "smoke-1", "--job", "gold-1"],
            ("attest", "local", ("smoke-1",), {"job": "gold-1"}),
        ),
        (
            ["stress", "gold", "--n", "1024", "--repetitions", "2"],
            ("stress", "gold", (), {"n": 1024, "repetitions": 2}),
        ),
        (
            ["provide", "serving", "--source", "compiled", "--expect", "d41d"],
            ("provide", "local", ("serving", "compiled", "d41d"), {}),
        ),
        (
            ["collect", "results/run", "--on", _MIYABI_G],
            (
                "fetch_path",
                "",
                (_MIYABI_G,),
                {"root": "/work/xg25g007/x10537/projects", "path": "results/run"},
            ),
        ),
    ],
    ids=[
        "run",
        "submit",
        "add",
        "remove",
        "upgrade",
        "new",
        "doctor",
        "install here",
        "install on a host",
        "setup",
        "setup sync-only",
        "sync",
        "shell",
        "serve",
        "interact",
        "compute",
        "monitor",
        "facts",
        "wait",
        "batch wait",
        "verdict",
        "cancel",
        "logs",
        "attest",
        "attest a named job",
        "stress",
        "provide",
        "collect from the profile's root",
    ],
)
def test_every_verb_reaches_the_board_method_it_names_with_the_flags_it_translated(
    depot: Path,
    relayed: list[Relayed],
    argv: list[str],
    expected: Relayed,
) -> None:
    """A verb owns only its dispatch and its flag translation.

    The CLI is a dispatch table, so which method a verb reaches and what it turned its flags
    into is the whole of what belongs to it. Everything past that seam is tested where it
    lives.
    """
    with pytest.raises(SystemExit, match="0"):
        build(depot)(argv)
    assert relayed == [expected]


def test_a_verdict_with_no_rows_says_why_on_stderr_rather_than_printing_a_bare_heading(
    depot: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A silent empty table reads as a failure, so the note goes where it cannot be mistaken.

    stderr rather than stdout, so a machine-readable mode stays exactly what it was.
    """
    note = "evidence.jsonl holds 3 line(s), none of..."
    empty = StreamVerdict(stream="s", trials=(), note=note)
    monkeypatch.setattr(Verdicts, "of", lambda self, target, host="", run="": empty)
    with pytest.raises(SystemExit, match="3"):
        build(depot)(["verdict", "s", "--json"])
    printed = capsys.readouterr()
    assert note in printed.err and note not in printed.out


@pytest.mark.parametrize(
    ("captured", "code", "shown"),
    [
        pytest.param("epoch 1\nepoch 2\n", 0, "epoch 2", id="a-run-whose-output-came-home"),
        pytest.param(
            "   \n", 1, "no output on file", id="a-run-nothing-was-ever-dispatched-or-captured"
        ),
        pytest.param(
            "\x1b[1mI\x1b[0m| epoch 2\n", 0, "I| epoch 2", id="a-coloured-log-read-off-a-terminal"
        ),
    ],
)
def test_the_logs_verb_prints_what_a_job_printed_or_says_nothing_was_kept(
    depot: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    captured: str,
    code: int,
    shown: str,
) -> None:
    """A script has to tell an empty log from a missing one, so the two exit differently."""
    monkeypatch.setattr(Verdicts, "captured", lambda self, handle, host="": captured)
    with pytest.raises(SystemExit, match=str(code)):
        build(depot)(["logs", "4242"])
    printed = capsys.readouterr()
    assert shown in (printed.out + printed.err)
    assert "\x1b" not in printed.out


@pytest.mark.parametrize(
    ("polled", "code", "shown"),
    [
        pytest.param(
            JobState(
                handle="3289319",
                state="Q",
                verdict="queued",
                note="estimated start Thu Sep  4 14:00:00 2026",
            ),
            2,
            (
                f"3289319 is queued on {_MIYABI_G}",
                "scheduler state Q",
                "submitted 2026-09-04T09:12:04+00:00",
                "estimated start Thu Sep  4 14:00:00 2026",
            ),
            id="queued-behind-a-full-cluster",
        ),
        pytest.param(
            JobState(handle="3289319", state="F", exit_code=1, verdict="failed"),
            1,
            ("no output on file for 3289319",),
            id="already-settled-by-its-scheduler",
        ),
    ],
)
def test_an_empty_log_says_where_its_job_stands_and_exits_on_whether_it_still_might_print(
    depot: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    polled: JobState,
    code: int,
    shown: tuple[str, ...],
) -> None:
    """The empty log of a job queued behind a full cluster read exactly like a silent job's.

    A job its scheduler has already settled will never print, so it gets the plain absence and
    exit 1, while one that has not started exits 2 with where it stands.
    """
    monkeypatch.setattr(Verdicts, "captured", lambda self, handle, host="": "")
    Cache(depot / db_file()).record(
        RunRecord(
            handle="3289319",
            target=_MIYABI_G,
            kind="pbs",
            script="job.sh",
            args="",
            git_sha="abc1234",
            dirty=0,
            submitted_at="2026-09-04T09:12:04+00:00",
        )
    )
    monkeypatch.setattr(
        Board, "job", lambda self, handle, host="": SimpleNamespace(poll=lambda: polled)
    )
    with pytest.raises(SystemExit, match=f"^{code}$"):
        build(depot)(["logs", "3289319"])
    err = capsys.readouterr().err
    assert all(fragment in err for fragment in shown)


@pytest.mark.parametrize(
    "argv",
    [
        pytest.param(
            ["submit", "--on", _MIYABI_G, "--walltim", "01:00:00", "--", "python", "x"],
            id="a-misspelled-option-before-the-delimiter",
        ),
        pytest.param(
            ["run", "--on", "gold", "--enve", "serving", "--", "true"],
            id="an-option-this-cli-never-had",
        ),
    ],
)
def test_an_option_this_cli_does_not_know_is_refused_by_name_not_folded_into_the_command(
    depot: Path, relayed: Sequence[Relayed], argv: list[str]
) -> None:
    """An unknown flag used to be appended to the user's command and fail on the host minutes
    later, which cost a campaign four jobs. It is a parse-time refusal naming the option instead.
    """
    with pytest.raises(SystemExit) as refused:
        build(depot)(argv)
    assert refused.value.code != 0
    assert relayed == []


@pytest.mark.parametrize("flag", ["--version", "--help", "-h"], ids=["--version", "--help", "-h"])
def test_the_passthrough_verbs_hand_the_clis_own_flags_to_the_command_after_the_delimiter(
    depot: Path,
    relayed: Sequence[Relayed],
    flag: str,
) -> None:
    """The passthrough verbs give the version flag up entirely.

    `--version` and `--help` after `--` belong to the wrapped program, not to this CLI.
    """
    with pytest.raises(SystemExit, match="0"):
        build(depot)(["run", "--", "python", flag])
    with pytest.raises(SystemExit, match="0"):
        build(depot)(["submit", "--on", _MIYABI_G, "--", "python", flag])
    assert [call[0] for call in relayed] == ["run", "submit"]
    assert [call[2] for call in relayed] == [
        (("python", flag),),
        (f"python {flag}",),
    ]


def test_a_passthrough_verb_still_documents_itself_before_the_delimiter(
    depot: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Dropping the version flag from the passthrough verbs must not cost them their help."""
    with pytest.raises(SystemExit, match="0"):
        build(depot)(["run", "--help"])
    assert "container" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("flags", "expected"),
    [
        ([], "4242"),
        (["--agent"], _FIELD_VALUE_HEADER),
        (["--json"], ""),
    ],
    ids=["the bare id a shell captures", "the compact record", "the whole handle as json"],
)
def test_submit_prints_the_bare_handle_unless_a_record_was_asked_for(
    depot: Path,
    relayed: Sequence[Relayed],
    capsys: pytest.CaptureFixture[str],
    flags: list[str],
    expected: str,
) -> None:
    with pytest.raises(SystemExit, match="0"):
        build(depot)(["submit", "--on", _MIYABI_G, *flags, "true"])
    out = capsys.readouterr().out
    if not expected:
        assert json.loads(out) == {
            "id": "4242",
            "host": _MIYABI_G,
            "root": "/work/p",
            "kind": "pbs",
            "fetch_path": None,
        }
        return
    assert out.splitlines()[0] == expected


@pytest.mark.parametrize(
    ("priced", "answer", "yes", "dispatched", "said"),
    [
        (
            JobEstimate(
                job="j",
                target="vast",
                kind="vast",
                hardware="1x RTX 4090",
                rate_usd_hr=0.31,
                expected_usd=0.05,
                p90_usd=0.08,
                rate_source="live",
            ),
            "n",
            False,
            False,
            "$0.31/hr (live)",
        ),
        (
            JobEstimate(job="j", target=_MIYABI_G, kind="pbs", rate_source="owned"),
            "y",
            False,
            True,
            "owned, expected $0.00",
        ),
        (
            JobEstimate(job="j", target=_MIYABI_G, kind="pbs", rate_source="owned"),
            "never asked",
            True,
            True,
            "owned, expected $0.00",
        ),
    ],
    ids=["declined, a rented target priced", "confirmed, owned hardware", "--yes skips the ask"],
)
def test_submit_prints_the_expectation_and_asks_once_at_a_terminal(
    depot: Path,
    relayed: Sequence[Relayed],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    priced: JobEstimate,
    answer: str,
    yes: bool,
    dispatched: bool,
    said: str,
) -> None:
    """The cost line prints before anything moves, and a declined dispatch never leaves.

    The line goes to stderr so the handle on stdout stays a shell's to capture, a rented
    target shows the meter and its tail, owned hardware says so instead of a hollow zero, and
    `--yes` is the script's way past the one question a terminal gets asked. The question itself
    is on stderr for the same reason the line above it is.
    """
    monkeypatch.setattr(Board, "expectation", lambda self, command, **query: priced)
    monkeypatch.setattr("sys.stdin", SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr("builtins.input", lambda: answer)
    argv = ["submit", "--on", _MIYABI_G, *(["--yes"] if yes else []), "true"]
    with pytest.raises(SystemExit, match="0" if dispatched else "1"):
        build(depot)(argv)
    printed = capsys.readouterr()
    assert said in printed.err
    assert not printed.out.startswith("dispatch?")
    assert [call[0] for call in relayed] == (["submit"] if dispatched else [])


def test_the_submit_expectation_names_what_comes_home_before_anything_moves(
    depot: Path,
    relayed: Sequence[Relayed],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A dispatch pulling nothing back is cheap to notice now and expensive to notice later.

    The node names its own evidence, so a `--node` submit says which directory comes home; a
    submit that names neither says that nothing does, in the one moment before a wave writes its
    receipts onto a cluster nobody is going to rsync by hand.
    """
    priced = JobEstimate(job="j", target=_MIYABI_G, kind="pbs", rate_source="owned")
    monkeypatch.setattr(Board, "expectation", lambda self, command, **query: priced)
    node = depot / "experiments" / "recovery_cost_cards" / "evidence"
    node.mkdir(parents=True)
    try:
        with pytest.raises(SystemExit, match="0"):
            build(depot)(
                ["submit", "--on", _MIYABI_G, "--yes", "--node", "recovery_cost_cards", "true"]
            )
        named = capsys.readouterr().err
        with pytest.raises(SystemExit, match="0"):
            build(depot)(["submit", "--on", _MIYABI_G, "--yes", "true"])
        silent = capsys.readouterr().err
    finally:
        rmtree(depot / "experiments", ignore_errors=True)

    assert "results experiments/recovery_cost_cards/evidence" in named
    assert "results NOT pulled back (no --fetch, no --node)" in silent


@pytest.mark.parametrize(
    ("argv", "code", "reached"),
    [
        pytest.param(
            ["run", "pytest", "--noconftest", "-x"],
            "0",
            [("run", "local", (("pytest", "--noconftest", "-x"),), {"env": "", "container": ""})],
            id="run-a-command-whose-flags-this-tool-never-had",
        ),
        pytest.param(
            ["run", "--env", "serving", "python", "-", "--", "-q"],
            "0",
            [
                (
                    "run",
                    "local",
                    (("python", "-", "--", "-q"),),
                    {"env": "serving", "container": ""},
                )
            ],
            id="run-stdin-python-its-own-delimiter-kept",
        ),
        pytest.param(
            ["submit", "--on", _MIYABI_G, "--yes", "python", "train.py", "--epochs", "3"],
            "0",
            [("submit", _MIYABI_G, ("python train.py --epochs 3",), {**_RESOURCES})],
            id="submit-a-command-with-flags",
        ),
        pytest.param(
            ["submit", "--on", _MIYABI_G, "--walltim", "01:00:00", "python", "x"],
            "1",
            [],
            id="a-misspelled-option-before-the-command-is-refused",
        ),
    ],
)
def test_the_entry_point_hands_everything_from_the_command_on_to_the_command(
    depot: Path,
    relayed: Sequence[Relayed],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    argv: list[str],
    code: str,
    reached: list[Relayed],
) -> None:
    """`run pytest --noconftest` is `uv run`'s grammar, and a typo in this tool's options is not.

    The snapshot is brought current before any of it, and nothing about that reaches stdout.
    """
    refreshed: list[str] = []
    monkeypatch.setattr("mainboard.cli.staleness.current", lambda: refreshed.append("current"))
    monkeypatch.setattr("sys.argv", ["mainboard", *argv])
    with pytest.raises(SystemExit, match=code):
        main()
    assert refreshed == ["current"]
    assert relayed == reached
    if code != "0":
        assert "--walltim" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("flags", "fragments"),
    [
        (["--json"], ()),
        ([], ("miyabi-g", "plan")),
        (["--agent"], (_FIELD_VALUE_HEADER, _MIYABI_G)),
    ],
    ids=["as json", "as the default rich table", "as the compact record"],
)
def test_the_plan_verb_prints_the_resolved_plan(
    depot: Path, capsys: pytest.CaptureFixture[str], flags: list[str], fragments: tuple[str, ...]
) -> None:
    with pytest.raises(SystemExit, match="0"):
        build(depot)(["plan", _MIYABI_G, *flags])
    out = capsys.readouterr().out
    if not fragments:
        plan = json.loads(out)
        assert plan["host"] == _MIYABI_G
        assert plan["container"]["image"].startswith("nvcr.io")
        return
    assert all(fragment in out for fragment in fragments)


@pytest.mark.parametrize(
    ("flags", "fields"),
    [
        ([], {"workspace", "environments", "containers", "hosts", "tasks"}),
        (["--fields", "workspace, , hosts"], {"workspace", "hosts"}),
    ],
    ids=["the whole declared surface", "a projection that trims and drops blank entries"],
)
def test_the_check_verb_lists_what_the_manifest_declares(
    depot: Path, capsys: pytest.CaptureFixture[str], flags: list[str], fields: set[str]
) -> None:
    with pytest.raises(SystemExit, match="0"):
        build(depot)(["check", "--json", *flags])
    surface = json.loads(capsys.readouterr().out)
    assert set(surface) == fields
    assert surface["workspace"] == "lab"


def test_the_mode_flags_refuse_each_other_before_anything_is_probed(
    depot: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(self: Survey) -> list[ComputePath]:
        raise AssertionError("the mode flags are checked before any probe runs")

    monkeypatch.setattr(Survey, "paths", refuse)
    with pytest.raises(MissionError, match="only one"):
        build(depot)(["compute", "--json", "--agent"])


def test_lanes_refuses_a_windows_roster_before_collecting_or_dispatching(depot: Path) -> None:
    manifest = depot / "mainboard.toml"
    original = manifest.read_text()
    try:
        manifest.write_text(original + '\n[hosts.homelab]\nplatform = "win-64"\nkind = "ssh"\n')
        with pytest.raises(MissionError, match="no jobs were dispatched"):
            build(depot)(["lanes", "run", "missing.py::test", "--on", "gold,homelab", "--yes"])
    finally:
        manifest.write_text(original)


def test_the_compute_verb_prices_and_credits_the_provider_rows(
    depot: Path,
    relayed: Sequence[Relayed],
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit, match="0"):
        build(depot)(["compute", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert [row["name"] for row in payload] == ["local", _MIYABI_G, "vast"]
    assert payload[0]["access"] == "here"
    assert datetime.fromisoformat(payload[2]["observed_at"]).tzinfo is UTC
    assert payload[2] == {
        "name": "vast",
        "kind": "provider",
        "access": "keyed",
        "detail": "1x RTX 4090 Texas, US",
        "usd_hr": 0.31,
        "credit_usd": 42.5,
        "observed_at": payload[2]["observed_at"],
        "cached_at": "",
    }


@pytest.mark.parametrize(
    ("flags", "fragments"),
    [
        ([], ("2 running", "unreachable", "daemon down", "failed", "results/run")),
        (["--agent", "--fields", "running,changed"], (_FIELD_VALUE_HEADER, "running", "changed")),
    ],
    ids=["what moved this pass, one row each", "the whole document projected onto two fields"],
)
def test_the_monitor_verb_prints_what_moved_or_the_whole_report(
    depot: Path,
    relayed: Sequence[Relayed],
    capsys: pytest.CaptureFixture[str],
    flags: list[str],
    fragments: Sequence[str],
) -> None:
    """Each monitor mode serves its own reader.

    A cron reads the full report and branches on it, a person at a terminal wants the jobs
    that actually settled, so the compact modes carry the document and the table carries
    rows.
    """
    with pytest.raises(SystemExit, match="0"):
        build(depot)(["monitor", *flags])
    out = capsys.readouterr().out
    assert all(fragment in out for fragment in fragments)


def test_the_monitor_verb_carries_the_counts_and_the_changed_flag_in_json(
    depot: Path,
    relayed: Sequence[Relayed],
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit, match="0"):
        build(depot)(["monitor", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["running"] == 2
    assert payload["changed"] is True
    assert payload["finished"][0]["pulled_path"] == "results/run"
    assert payload["unreachable_hosts"][0]["host"] == _MIYABI_G


@pytest.mark.parametrize(
    ("flags", "expected"),
    [
        (
            ["--json"],
            {
                "running": 0,
                "resumed": [],
                "held": [],
                "finished": [],
                "failed": [],
                "unreachable_hosts": [],
            },
        ),
        ([], None),
    ],
    ids=["a quiet pass says exactly that", "a quiet pass still prints its heading"],
)
def test_the_monitor_verb_sweeps_an_untouched_cache_without_changes(
    depot: Path,
    capsys: pytest.CaptureFixture[str],
    flags: list[str],
    expected: dict[str, int | list[str]] | None,
) -> None:
    """An empty change table still names its columns.

    A reader sees a heading rather than nothing at all.
    """
    with pytest.raises(SystemExit, match="0"):
        build(depot)(["monitor", *flags])
    out = capsys.readouterr().out
    if expected is None:
        assert "monitor: 0 running" in out
        assert "outcome" in out
        return
    assert json.loads(out) == {**expected, "changed": False}


@pytest.mark.parametrize(
    "interrupted", [False, True], ids=["every pass renders", "an interrupt stops it quietly"]
)
def test_the_monitor_verb_watches_in_the_foreground_until_it_is_stopped(
    depot: Path,
    relayed: Sequence[Relayed],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    interrupted: bool,
) -> None:
    if interrupted:
        monkeypatch.setattr("mainboard.monitor.Monitor.watch", lambda self, interval: _interrupt())
    with pytest.raises(SystemExit, match="0"):
        build(depot)(["monitor", "--watch", "0.1", "--json"])
    assert capsys.readouterr().out.count('"running": 2') == (0 if interrupted else 2)


def _interrupt() -> None:
    raise KeyboardInterrupt


def test_a_machine_readable_verb_leaves_stdout_to_its_document_alone(
    depot: Path, capfd: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A `--json` consumer parses stdout whole, so nothing the work says may land in front of it.

    The two shapes that actually did land there are both here: a library greeting the terminal
    through this process, and a child process narrating a transfer straight onto the descriptor.
    """

    def noisy(self: Monitor) -> MonitorReport:
        print("wandb: [wandb.login()] Loaded credentials from netrc")
        os.write(1, b"sending incremental file list\n")
        return MonitorReport(running=1)

    monkeypatch.setattr(Monitor, "once", noisy)
    with pytest.raises(SystemExit, match="0"):
        build(depot)(["monitor", "--json"])
    printed = capfd.readouterr()
    assert json.loads(printed.out)["running"] == 1
    assert "wandb: [wandb.login()]" in printed.err
    assert "sending incremental file list" in printed.err


@pytest.mark.parametrize(
    ("argv", "code", "fragment"),
    [
        (["mainboard", "check"], "0", "lab"),
        (["mainboard", "plan", "gold", "--env", "ghost"], "1", "declared environments"),
    ],
    ids=["a clean verb from a directory below the root", "a refusal printed without a traceback"],
)
def test_the_entry_point_discovers_the_workspace_and_refuses_without_a_traceback(
    depot: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    argv: list[str],
    code: str,
    fragment: str,
) -> None:
    nested = depot / "deep" / "inside"
    nested.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(nested)
    monkeypatch.setattr("sys.argv", argv)
    with pytest.raises(SystemExit, match=code):
        main()
    printed = capsys.readouterr()
    assert fragment in (printed.out if code == "0" else printed.err)


def test_submit_carries_every_need_it_was_given_to_the_board(
    depot: Path, relayed: Sequence[Relayed]
) -> None:
    """`--needs` repeats, and each one reaches the dispatch as the job file's own would."""
    with pytest.raises(SystemExit, match="0"):
        build(depot)(
            ["submit", "--on", _MIYABI_G, "--needs", "data/a", "--needs", "data/b", "true"]
        )
    [(verb, host, args, options)] = relayed
    assert (verb, host, args) == ("submit", _MIYABI_G, ("true",))
    assert options["needs"] == ("data/a", "data/b")


def test_collect_refuses_a_host_without_a_declared_root_before_contacting_it(
    depot: Path, relayed: Sequence[Relayed]
) -> None:
    """Without a root there is no remote path to read, and guessing one could publish the wrong
    tree, so the refusal names the manifest key to declare and nothing is fetched."""
    with pytest.raises(MissionError, match=r"declare hosts\.gold\.root"):
        build(depot)(["collect", "results/run", "--on", "gold"])
    assert relayed == []


def test_a_query_without_sql_reads_the_runs_view(tmp_path: Path) -> None:
    """The bare verb is the first question anyone asks of collected results: what ran."""
    implicit, explicit = tmp_path / "implicit.csv", tmp_path / "explicit.csv"
    with pytest.raises(SystemExit, match="^0$"):
        build(tmp_path)(["query", "--out", str(implicit)])
    with pytest.raises(SystemExit, match="^0$"):
        build(tmp_path)(["query", "SELECT * FROM runs", "--out", str(explicit)])
    assert implicit.read_text() == explicit.read_text()


@pytest.mark.parametrize(
    ("flags", "shown"),
    [([], str(Path("/envs/lab-4f2a"))), (["--json"], None)],
    ids=["the bare prefix a shell captures", "the prefix as json"],
)
def test_provide_prints_the_bare_prefix_unless_json_was_asked_for(
    depot: Path,
    relayed: Sequence[Relayed],
    capsys: pytest.CaptureFixture[str],
    flags: list[str],
    shown: str | None,
) -> None:
    """A dispatched job activates whatever this line says, so the bare form is the path alone."""
    with pytest.raises(SystemExit, match="^0$"):
        build(depot)(["provide", *flags])
    out = capsys.readouterr().out
    if shown is None:
        assert json.loads(out) == {"prefix": str(Path("/envs/lab-4f2a"))}
        return
    assert out == f"{shown}\n"


@pytest.mark.parametrize("json_mode", [True, False], ids=["the report json", "the compact record"])
def test_stress_prints_the_whole_report_or_one_row_per_precision_and_link(
    depot: Path, relayed: Sequence[Relayed], capsys: pytest.CaptureFixture[str], json_mode: bool
) -> None:
    """The JSON is what a remote read parses back, so it carries every field; the table rounds
    and names why a precision was skipped rather than printing its zero bare."""
    with pytest.raises(SystemExit, match="^0$"):
        build(depot)(["stress", "--json" if json_mode else "--agent"])
    out = capsys.readouterr().out
    if json_mode:
        report = json.loads(out)
        assert (report["device"], report["datasheet_fp32_tflops"]) == ("GH200", 66.93)
        assert [rate["precision"] for rate in report["rates"]] == ["bf16", "fp8"]
        return
    assert all(
        fragment in out
        for fragment in ("GH200", "66.9", "BF16", "687.3", "no FP8 kernels", "412.1")
    )


# One busy card on this machine, and the first line of why gold could not be read, which is all
# of a failure a table cell has room for; the rest of the message must stay out of the table.
_READING = Occupancy(
    hostname="box",
    at="2026-09-25T00:00:00+00:00",
    cards=(
        CardOccupancy(
            index=0,
            name="RTX 4090",
            utilization_pct=97,
            memory_used_bytes=20_000_000_000,
            memory_total_bytes=24_000_000_000,
            holders=(Holder(pid=4242, user="pedro", age_s=7200, command="python train.py"),),
        ),
    ),
)
_WHY = "ssh: connect to host gold port 22: no route"
_DETAIL = "the remote traceback nobody reads in a table"


def occupied(failure: type[Exception] = MissionError) -> Callable[[Board], Occupancy]:
    """`Board.occupancy` answering `_READING` here and failing with `failure` on gold."""

    def read(self: Board) -> Occupancy:
        if self.host == "gold":
            raise failure(f"{_WHY}\n{_DETAIL}")
        return _READING

    return read


@pytest.mark.parametrize(
    ("argv", "expected", "indent"),
    [
        (["gpus", "--json"], _READING.model_dump(mode="json"), None),
        (["gpus", "gold", "--json"], {}, None),
        (["gpus", "--every", "--json"], {"local": _READING.model_dump(mode="json")}, 2),
    ],
    ids=[
        "one host prints its reading alone on one line",
        "one unreachable host prints an empty reading",
        "every host keys readings by name and leaves the unreachable out",
    ],
)
def test_the_gpus_verb_prints_json_a_remote_read_parses_back(
    depot: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    argv: list[str],
    expected: dict[str, JsonValue],
    indent: int | None,
) -> None:
    """A single host's reading is one line, which is exactly what `--every` parses off each
    ssh host, so it must never be wrapped in the fleet's keyed document."""
    monkeypatch.setattr(Board, "occupancy", occupied())
    with pytest.raises(SystemExit, match="^0$"):
        build(depot)(argv)
    assert capsys.readouterr().out == json.dumps(expected, indent=indent) + "\n"


@pytest.mark.parametrize("failure", [MissionError, OSError, ValueError])
def test_a_host_that_cannot_be_read_costs_the_fleet_table_one_row_and_nothing_else(
    depot: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    failure: type[Exception],
) -> None:
    """The screen exists to say which card is free, so one host down must not blank the rest,
    whether the read failed in ssh, in the tool, or in parsing what came back."""
    monkeypatch.setattr(Board, "occupancy", occupied(failure))
    with pytest.raises(SystemExit, match="^0$"):
        build(depot)(["gpus", "--every", "--agent"])
    out = capsys.readouterr().out
    assert "0: RTX 4090" in out
    assert "4242 pedro 2h python train.py" in out
    assert f"unreachable: {_WHY}" in out
    assert _DETAIL not in out


# A lane under a node directory, so its receipts serve that node, and the cells it collects:
# two models, one of them with two seeds, which `--group model` turns into two jobs.
_LANE = "experiments/sweep/test_lane.py::test_rate"
_CELLS = (("a-1", "a"), ("b-1", "b"), ("a-2", "a"))


def _collected() -> str:
    """What the collection prints: its `CELL` lines among the noise a pytest run makes."""
    lines = [
        "CELL "
        + Cell(nodeid=f"{_LANE}[{key}]", key=key, params={"model": model}).model_dump_json()
        for key, model in _CELLS
    ]
    return "\n".join(["collecting ...", *lines, "3 tests collected"])


@pytest.mark.parametrize(
    ("printed", "flags", "refusal", "match"),
    [
        ("collecting ...\nno tests ran\n", ["--yes"], MissionError, "collected no cells"),
        (_collected(), ["--dry-run"], SystemExit, "^0$"),
        (_collected(), [], SystemExit, "^1$"),
    ],
    ids=[
        "a lane that collects nothing is refused",
        "a dry run prints the plan and stops, never asking",
        "a declined terminal confirmation stops before any host",
    ],
)
def test_lanes_run_dispatches_nothing_until_a_plan_exists_and_is_agreed(
    depot: Path,
    relayed: Sequence[Relayed],
    launcher: Launcher,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    printed: str,
    flags: list[str],
    refusal: type[BaseException],
    match: str,
) -> None:
    """The cells come from a collection inside the workspace environment, never this process,
    since the lane imports what that environment holds; and the plan prints before anything
    moves, so a person at a terminal sees what one `y` would dispatch."""
    launcher.printed = printed
    monkeypatch.setattr("sys.stdin", SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr("builtins.input", lambda: "n")
    with pytest.raises(refusal, match=match):
        build(depot)(["lanes", "run", _LANE, "--on", "local,gold", "--group", "model", *flags])
    assert launcher.argv == [
        Project().name,
        *("run", "--", "python", "-m", "mainboard.jobs.lanes", "collect", _LANE),
    ]
    assert relayed == []
    if refusal is SystemExit:
        assert "a-1 a-2" in capsys.readouterr().out


@pytest.mark.parametrize("wait", [False, True], ids=["dispatch and return", "--wait settles"])
def test_lanes_run_runs_local_groups_in_place_and_submits_each_group_to_every_other_host(
    depot: Path,
    relayed: Sequence[Relayed],
    launcher: Launcher,
    capsys: pytest.CaptureFixture[str],
    wait: bool,
) -> None:
    """One group is one job: `local` runs it in this process's board, any other host gets a
    submission serving the lane's node, and `--wait` settles only what was submitted, since a
    local group has already finished by the time the loop reaches it."""
    launcher.printed = _collected()
    argv = ["lanes", "run", _LANE, "--on", "local,gold", "--group", "model", "--timeout", "60"]
    with pytest.raises(SystemExit, match="^0$"):
        build(depot)([*argv, "--rerun", "--yes", *(["--wait"] if wait else [])])

    def line(*ids: str) -> list[str]:
        fresh = ["--fresh", "--timeout", "60.0"]
        return [
            _LANE,
            "--",
            *fresh,
            *ids,
            "--",
            "-p",
            "no:randomly",
            "-q",
            "--no-header",
            "--rerun",
        ]

    resources = {"queue": "", "walltime": "", "mem_gb": 0, "gpus": 0, "gpu_name": ""}
    submitted = {**resources, "max_usd": 0.0, "node": "sweep"}
    assert relayed == [
        ("run", "local", (line("a-1", "a-2"),), {}),
        ("run", "local", (line("b-1"),), {}),
        ("submit", "gold", (" ".join(line("a-1", "a-2")),), {"name": "lanes-gold-a", **submitted}),
        ("submit", "gold", (" ".join(line("b-1")),), {"name": "lanes-gold-b", **submitted}),
        *([("wait", "", ("4242",), {"host": "gold"})] * (2 if wait else 0)),
    ]
    assert "local exit 0" in capsys.readouterr().out
