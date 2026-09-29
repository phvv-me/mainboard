"""Every everyday verb, run for real in a fresh workspace: it answers, or says why it can't."""

import json
import sys

import pytest

from .conftest import MB


@pytest.mark.parametrize(
    "args",
    [
        ["--help"],
        ["--version"],
        ["host", "--help"],
        ["job", "--help"],
        ["self", "--help"],
        ["doctor"],
        ["job", "list"],
        ["host", "list", "--facts"],
        ["self", "update"],
    ],
    ids=lambda args: " ".join(args),
)
def test_a_verb_answers_in_a_fresh_workspace(mb, args: list[str]) -> None:
    assert mb(*args).code == 0


def test_the_help_names_the_groups_and_pixis_environment_verbs(mb) -> None:
    listed = {line.strip().split(":")[0] for line in mb("--help").out.splitlines()}
    for verb in ("install", "lock", "add", "remove", "run", "shell", "host", "job", "git"):
        assert verb in listed, verb
    gone = {"activate", "check", "compute", "center", "jobs", "plot", "shell-hook", "submit"}
    assert not gone & listed


def test_a_bad_query_is_said_without_a_traceback(mb) -> None:
    ran = mb("query", "SELECT * FROM lake.nothing_here")
    assert ran.code == 1 and "nothing_here" in ran.err


def test_the_lake_is_queryable_as_soon_as_the_workspace_exists(mb) -> None:
    ran = mb("query", "SELECT count(*) AS n FROM lake.runs", "--json")
    assert ran.code == 0
    assert json.loads(ran.out) == [{"n": 0}]


def test_a_query_exports_through_duckdb(mb, workspace) -> None:
    target = workspace / "runs.csv"
    assert mb("query", "SELECT 1 AS one", "--out", str(target)).code == 0
    assert target.read_text(encoding="utf-8").splitlines() == ["one", "1"]


def test_an_environment_never_installed_is_named_not_crashed_on(mb) -> None:
    ran = mb("shell")
    assert ran.code == 1
    assert "no default environment" in ran.err and "mb install" in ran.err


def test_an_explicit_delimiter_is_respected(mb) -> None:
    ran = mb("proc", "timeout", "30", "--", sys.executable, "-c", "print('reached')")
    assert ran.code == 0 and "reached" in ran.out


def test_a_bounded_command_is_stopped_at_its_bound(mb) -> None:
    ran = mb("proc", "timeout", "1", sys.executable, "-c", "import time; time.sleep(30)")
    assert ran.code == 124


def test_a_workspace_is_required_and_said_to_be(mb, tmp_path_factory) -> None:
    elsewhere = tmp_path_factory.mktemp("nowhere")
    ran = mb("doctor", cwd=elsewhere)
    assert ran.code != 0
    assert "mb.toml" in ran.said or "mainboard.toml" in ran.said


@pytest.mark.skipif(sys.platform == "win32", reason="a pseudo-terminal needs POSIX")
def test_a_real_terminal_gets_the_same_answers(workspace) -> None:
    """The console renderer runs only on a terminal; it crashed there once while pipes passed."""
    import os
    import pty
    import subprocess

    primary, secondary = pty.openpty()
    done = subprocess.run(
        [MB, "job", "list"],
        cwd=workspace,
        stdin=secondary,
        stdout=secondary,
        stderr=secondary,
        timeout=120,
        check=False,
    )
    os.close(secondary)
    said = os.read(primary, 65536).decode(errors="replace")
    os.close(primary)
    assert done.returncode == 0, said
    assert "Traceback" not in said


def test_the_audit_reads_this_machine_and_names_every_fix(mb) -> None:
    ran = mb("host", "list", "--audit", "--json", timeout=600)
    assert ran.code == 0, ran.said
    rows = {row["section"]: row for row in json.loads(ran.out)}
    assert "system" in rows and "disk" in rows
    assert all(row["fix"] for row in rows.values() if row["verdict"] == "warn")


def test_an_upgrade_dry_run_only_prints_its_steps(mb) -> None:
    ran = mb("host", "upgrade", "--dry-run")
    assert ran.code == 0
    assert ran.out.startswith("$ ") and "self update" in ran.out


def test_runs_are_queryable_by_project_without_reading_json(mb) -> None:
    ran = mb("query", "SELECT project, name, verdict FROM lake.runs", "--json")
    assert ran.code == 0, ran.said
    assert json.loads(ran.out) == []
