"""Every everyday verb, run for real in a fresh workspace: it answers, or says why it can't."""

import json
import sys
import time

import pytest

from mainboard.cli import Delimiter
from mainboard.core import Section, Verdict
from mainboard.upkeep.machine import audit
from mainboard.upkeep.managers import Manager
from mainboard.verbs.constants import TREE

from ..constants import MB


@pytest.mark.parametrize(
    "args",
    [
        [],
        ["--help"],
        ["-h"],
        ["help", "data", "lake", "ingest"],
        ["--help", "ingest", "evidence"],
        ["--version"],
        ["--tree"],
        ["--tree", "data", "--all"],
        ["host", "--help"],
        ["job", "--help"],
        ["code"],
        ["code", "doctor"],
        ["job", "list"],
        ["host", "list", "--facts"],
        ["host", "update"],
    ],
    ids=lambda args: " ".join(args) or "the-bare-tool",
)
def test_a_verb_answers_in_a_fresh_workspace(mb, args: list[str]) -> None:
    assert mb(*args).code == 0


@pytest.mark.parametrize(
    ("args", "rows"),
    [
        pytest.param([], 10, id="the-ten-roots"),
        pytest.param(["code"], 8, id="a-group"),
        pytest.param(["data", "lake"], 9, id="a-group-two-words-deep"),
    ],
)
def test_a_bare_tool_or_group_lists_one_short_row_per_member(
    mb, args: list[str], rows: int
) -> None:
    """A header, a row per shown member in TREE's order, the way further, each in 75 columns."""
    lines = mb(*args, env={"MB_AGENT": "1"}).out.splitlines()
    entries = TREE.node(" ".join(args)).entries()
    members = [path[0] for path, entry in entries if len(path) == 1 and entry.show]
    assert [line.split()[0].rstrip("/") for line in lines[1:-1]] == members
    assert len(members) == rows
    assert max(map(len, lines)) <= 75
    assert lines[-1] == "mb <path> lists a group · add --help for arguments · mb --tree shows all"


@pytest.mark.parametrize("agent", ["1", "0"], ids=["an-agent", "a-person"])
def test_a_moved_command_is_refused_naming_its_new_path(mb, agent: str) -> None:
    ran = mb("lint", env={"MB_AGENT": agent})
    assert ran.code == 2
    assert "did you mean `mb code lint`?" in ran.err and "Unknown command" not in ran.err


def test_a_bad_query_is_said_without_a_traceback(mb) -> None:
    ran = mb("data", "query", "SELECT * FROM lake.nothing_here")
    assert ran.code == 1 and "nothing_here" in ran.err


def test_the_lake_is_queryable_as_soon_as_the_workspace_exists(mb) -> None:
    ran = mb("data", "query", "SELECT count(*) AS n FROM lake.runs", "--json")
    assert ran.code == 0
    assert json.loads(ran.out) == [{"n": 0}]


def test_a_query_exports_through_duckdb(mb, workspace) -> None:
    target = workspace / "runs.csv"
    assert mb("data", "query", "SELECT 1 AS one", "--out", str(target)).code == 0
    assert target.read_text(encoding="utf-8").splitlines() == ["one", "1"]


def test_an_environment_never_installed_is_named_not_crashed_on(mb) -> None:
    ran = mb("dep", "shell", "--as-is")
    assert ran.code == 1
    assert "no default environment" in ran.err and "mb install" in ran.err


def test_an_explicit_delimiter_is_respected(mb) -> None:
    ran = mb("proc", "timeout", "30", "--", sys.executable, "-c", "print('reached')")
    assert ran.code == 0 and "reached" in ran.out


def test_a_bounded_command_keeps_its_own_options_and_delimiter(mb) -> None:
    """`proc timeout 900 mb job submit --on gold -- ...` died on `--on`, read as this tool's."""
    said = "import sys; print(sys.argv[1:])"
    ran = mb("proc", "timeout", "30", sys.executable, "-c", said, "--on", "gold", "--", "-m", "x")
    assert ran.code == 0, ran.said
    assert "['--on', 'gold', '--', '-m', 'x']" in ran.out


@pytest.mark.parametrize(
    ("typed", "placed"),
    [
        (
            "proc timeout 900 mb job submit --on gold -- python x.py",
            "proc timeout 900 -- mb job submit --on gold -- python x.py",
        ),
        ("proc timeout 60 git log -- path", "proc timeout 60 -- git log -- path"),
        ("proc timeout 30 -- python x.py", "proc timeout 30 -- python x.py"),
        ("job submit --on gold -- python x.py", "job submit --on gold -- python x.py"),
        (
            "job submit --on gold python x.py --out y",
            "job submit --on gold -- python x.py --out y",
        ),
        ("run pytest x -- -k y", "run -- pytest x -- -k y"),
        ("run x.py::t -- --collect-only", "run -- x.py::t -- --collect-only"),
        # The verb's own options after the target, ended by the caller: left to the parser.
        ("job submit x.py::t --on gold -- --fresh a", "job submit x.py::t --on gold -- --fresh a"),
        ("run x.py --env lab -- -k y", "run x.py --env lab -- -k y"),
        # A program's own help, before its own `--`, is the program's.
        ("run tool --help -- x", "run -- tool --help -- x"),
    ],
)
def test_the_delimiter_goes_where_the_command_starts(typed: str, placed: str) -> None:
    assert Delimiter(TREE.cli()).placed(typed.split()) == placed.split()


def test_a_bounded_command_is_stopped_at_its_bound(mb) -> None:
    ran = mb("proc", "timeout", "1", sys.executable, "-c", "import time; time.sleep(30)")
    assert ran.code == 124


def test_a_workspace_is_required_and_said_to_be(mb, tmp_path_factory) -> None:
    elsewhere = tmp_path_factory.mktemp("nowhere")
    ran = mb("code", "doctor", cwd=elsewhere)
    assert ran.code != 0
    assert "mb.toml" in ran.said or "mainboard.toml" in ran.said


def test_a_real_terminal_gets_the_same_answers(at_a_terminal) -> None:
    """The console renderer runs only on a terminal; it crashed there once while pipes passed."""
    ran = at_a_terminal([MB, "job", "list"])
    assert ran.code == 0, ran.said
    assert "Traceback" not in ran.said


def test_the_audit_reads_this_machine_and_names_every_fix(mb) -> None:
    ran = mb("host", "list", "--audit", "--json", timeout=600)
    assert ran.code == 0, ran.said
    rows = {row["section"]: row for row in json.loads(ran.out)}
    assert "system" in rows and "disk" in rows
    assert all(row["fix"] for row in rows.values() if row["verdict"] == "warn")


def test_a_manager_that_never_answers_is_unknown_instead_of_waited_for() -> None:
    """`snap` without snapd's socket retries for two minutes; the audit gives each manager 30 s."""
    stuck = Manager(
        name="stuck",
        program="sleep",
        pending=("sleep", "60"),
        count=len,
        steps=("true",),
        timeout=0.5,
    )
    started = time.monotonic()
    rows = {row.section: row for row in audit(managers=(stuck,))}
    assert time.monotonic() - started < 30
    assert rows["stuck"] == Section(
        section="stuck", verdict=Verdict.WARN, detail="unknown, no answer in 0.5 s", fix="sleep 60"
    )


def test_an_upgrade_dry_run_only_prints_its_steps(mb) -> None:
    ran = mb("host", "upgrade", "--dry-run")
    assert ran.code == 0
    assert ran.out.startswith("$ ") and "host update" in ran.out


def test_runs_are_queryable_by_project_without_reading_json(mb) -> None:
    ran = mb("data", "query", "SELECT project, name, verdict FROM lake.runs", "--json")
    assert ran.code == 0, ran.said
    assert json.loads(ran.out) == []


def test_an_environment_narrowing_the_mirror_still_ships_every_lock_input(mb, workspace) -> None:
    """The lock's digest reads every local project's metadata, so a host must receive each one."""
    (workspace / "mb.toml").write_text(
        "\n".join(
            (
                "[workspace]",
                'name = "it"',
                "[python.deps]",
                'lib = { path = "packages/lib" }',
                "[envs.lean]",
                "no-default = true",
                'sources = ["src"]',
                "[hosts.box]",
                'kind = "ssh"',
                "",
            )
        ),
        encoding="utf-8",
        newline="\n",
    )
    ran = mb("host", "list", "box", "--plan", "--env", "lean")
    assert ran.code == 0, ran.said
    assert '"src"' in ran.out and "packages/lib/pyproject.toml" in ran.out
