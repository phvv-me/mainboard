import json
from collections.abc import Sequence
from typing import TYPE_CHECKING

import pytest

from mainboard import ComputePath, MissionError, Survey
from mainboard.cli import build, main

if TYPE_CHECKING:
    from pathlib import Path

    from .support import Relayed

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
    "env": "",
    "container": "",
}


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (
            ["run", "--on", "gold", "--env", "serving", "--", "python", "-c", "print(1)"],
            ("run", "gold", ("python -c 'print(1)'",), {"env": "serving", "container": ""}),
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
            ("install", "gold", ("serving",), {"resolve": False}),
        ),
        (["shell", "--env", "serving"], ("shell", "local", ("serving",), {})),
        (
            ["interact", "--on", "gold", "--queue", "interact-g", "--", "pwd"],
            ("interact", "gold", ("pwd",), {"env": "", "queue": "interact-g", "walltime": ""}),
        ),
        (["compute"], ("paths", "", (), {})),
        (["monitor"], ("once", "", (), {})),
        (["facts", "gold"], ("facts", "gold", (), {})),
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
        "shell",
        "interact",
        "compute",
        "monitor",
        "facts",
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
    assert [call[2] for call in relayed] == [(f"python {flag}",), (f"python {flag}",)]


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
    assert payload[2] == {
        "name": "vast",
        "kind": "provider",
        "access": "keyed",
        "detail": "1x RTX 4090 Texas, US",
        "usd_hr": 0.31,
        "credit_usd": 42.5,
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
        (["--json"], {"running": 0, "finished": [], "failed": [], "unreachable_hosts": []}),
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
