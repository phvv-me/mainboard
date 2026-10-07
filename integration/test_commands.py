"""Every everyday verb, run for real in a fresh workspace: it answers, or says why it can't."""

import json
import os
import shutil
import subprocess
import sys
from decimal import Decimal

import duckdb
import polars as pl
import pytest

from mainboard.cli import build
from mainboard.core.errors import MissionError
from mainboard.delimiter import Delimiter
from mainboard.dispatch import commandline
from mainboard.results import Results

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


def test_dataframe_queries_preserve_types_without_pyarrow(workspace, monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "pyarrow", None)
    results = Results(workspace)
    frame = results.query("""
        SELECT i::TINYINT AS x,
               CASE WHEN i = 2 THEN NULL ELSE i / 10 END::DECIMAL(4,2) AS y,
               [struct_pack(n := i::SMALLINT)] AS nested
        FROM range(4) t(i) ORDER BY i DESC
    """)
    assert frame.schema == {
        "x": pl.Int8,
        "y": pl.Decimal(4, 2),
        "nested": pl.List(pl.Struct({"n": pl.Int16})),
    }
    # Results.query has already closed the DuckDB connection; these buffers remain owned.
    assert frame.to_dicts() == [
        {"x": i, "y": None if i == 2 else Decimal(i) / 10, "nested": [{"n": i}]}
        for i in (3, 2, 1, 0)
    ]
    empty = results.query("""
        SELECT NULL::SMALLINT AS x, []::STRUCT(n TINYINT)[] AS nested WHERE false
    """)
    assert empty.is_empty()
    assert empty.schema == {"x": pl.Int16, "nested": pl.List(pl.Struct({"n": pl.Int8}))}
    with pytest.raises(MissionError, match="not_a_column"):
        results.query("SELECT not_a_column")
    with pytest.raises(MissionError, match="Could not convert"):
        results.query("SELECT CAST('bad' AS INTEGER)")
    with pytest.raises(ValueError, match="one SELECT"):
        results.query("SELECT 1; SELECT 2")


def test_dataframe_queries_execute_once(workspace, monkeypatch) -> None:
    results = Results(workspace)
    original = results._views

    def views(connection: duckdb.DuckDBPyConnection, project: str, sql: str) -> None:
        connection.execute("CREATE SEQUENCE query_counter")
        original(connection, project, sql)

    monkeypatch.setattr(results, "_views", views)
    frame = results.query("SELECT nextval('query_counter') AS n FROM range(4)")
    assert frame["n"].to_list() == [1, 2, 3, 4]


def test_an_environment_never_installed_is_named_not_crashed_on(mb) -> None:
    ran = mb("shell")
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
    ],
)
def test_the_delimiter_goes_where_the_command_starts(typed: str, placed: str) -> None:
    assert Delimiter(build()).placed(typed.split()) == placed.split()


def test_a_path_the_shell_rewrote_is_refused_naming_the_switch(monkeypatch) -> None:
    """Git Bash turned `--out-dir /home/crimson/y` into a path under its own folder, and a GPU
    job wrote an hour of results under that literal name on a Linux host."""
    monkeypatch.setattr(commandline, "_shell_roots", lambda: ("C:/Program Files/Git/",))
    rewritten = r"C:\Program Files\Git\home\crimson\y"
    with pytest.raises(MissionError, match="MSYS_NO_PATHCONV=1") as refused:
        commandline.joined(["python", "-m", "x", "--out-dir", rewritten])
    assert "`/home/crimson/y`" in str(refused.value)
    with pytest.raises(MissionError, match="`--out=/home/y`"):
        commandline.joined(["python", "x.py", "--out=C:/Program Files/Git/home/y"])
    assert commandline.joined(["python", "x.py", "D:/data/y"]) == "python x.py D:/data/y"


@pytest.mark.skipif(
    sys.platform != "win32"
    or shutil.which("cygpath") is None
    or "MSYS_NO_PATHCONV" in os.environ
    or os.environ.get("MSYS2_ARG_CONV_EXCL") == "*",
    reason="needs an MSYS shell on PATH with its path conversion on",
)
def test_a_submit_refuses_a_rewritten_path_before_any_host_is_asked(mb) -> None:
    root = subprocess.run(
        ["cygpath", "-m", "/"], capture_output=True, text=True, encoding="utf-8", check=True
    ).stdout.strip()
    typed = ["job", "submit", "--on", "nowhere", "--", "python", "x.py"]
    ran = mb(*typed, f"{root.rstrip('/')}/home/me/out", env={"MSYSTEM": "MINGW64"})
    assert ran.code == 1
    assert "MSYS_NO_PATHCONV=1" in ran.err and "`/home/me/out`" in ran.err


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
    import pty

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
