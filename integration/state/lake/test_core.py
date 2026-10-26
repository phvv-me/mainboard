import gc
import json
import os
import socket
import sqlite3
import subprocess
import sys
import weakref
from collections.abc import Iterator, Sequence
from contextlib import ExitStack, closing
from datetime import UTC, datetime
from pathlib import Path

import pytest

from mainboard.state import Lake, schema

from ...constants import MB

_OLDER_SPEC = "UPDATE ducklake_metadata SET value = '1.1-dev1' WHERE key = 'version'"
_WRITE_FROM_FAR = """from datetime import UTC, datetime
from pathlib import Path
from sqlalchemy import func, select
from mb.state import schema
from mb.state.lake import Lake
session = Lake.at(Path.cwd()).session()
log = schema.schema_log
session.append(log, [{'ts': datetime.now(UTC), 'version': 0, 'spec': 'from-far', 'engine': 'it'}])
print(session.rows(select(func.count()).where(log.c.spec == 'from-far')))
sql = 'SELECT count(*) FROM lake.schema_log WHERE spec = ?'
print(session.rows(sql, ['from-far']))
"""


def _nested(workspace: Path) -> Path:
    """A project inside the workspace, tracked by the git repository the workspace becomes."""
    project = workspace / "research" / "lab"
    project.mkdir(parents=True)
    (project / "mb.toml").write_text('[workspace]\nname = "lab"\n', encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(workspace)], check=True)
    subprocess.run(["git", "-C", str(workspace), "add", "mb.toml", "research"], check=True)
    return project


def _downgraded(lake: Lake) -> None:
    """Record the lake's catalog at an older DuckLake spec."""
    with closing(sqlite3.connect(lake.catalog)) as catalog, catalog:
        catalog.execute(_OLDER_SPEC)


def _is_listening(workspace: Path, port: int) -> bool:
    """Whether something listened on `port` within ninety seconds."""
    waited = subprocess.run(
        [MB, "proc", "wait", "--port", f"localhost:{port}", "--timeout", "90"],
        cwd=workspace,
        check=False,
    )
    return waited.returncode == 0


def _stop(server: subprocess.Popen[str]) -> None:
    subprocess.run([MB, "proc", "kill", str(server.pid)], check=False)
    server.wait(timeout=60)


@pytest.fixture
def lakes(workspace: Path) -> list[Lake]:
    """The lakes of the workspace and of a project inside it, both created."""
    found = [Lake.at(workspace), Lake.at(_nested(workspace))]
    for lake in found:
        lake.create()
    return found


@pytest.fixture
def elsewhere(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A second workspace with no lake of its own."""
    folder = tmp_path_factory.mktemp("elsewhere")
    (folder / "mb.toml").write_text('[workspace]\nname = "far"\n', encoding="utf-8")
    return folder


@pytest.fixture
def served(workspace: Path) -> Iterator[dict[str, str]]:
    """The workspace's lake served on a free port, and the environment reaching it."""
    with socket.socket() as probe:
        probe.bind(("localhost", 0))
        port = probe.getsockname()[1]
    with ExitStack() as cleanup:
        server = subprocess.Popen(
            [MB, "lake", "serve", "--port", str(port)],
            cwd=workspace,
            env={**os.environ, "NO_COLOR": "1"},
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        cleanup.callback(_stop, server)
        assert _is_listening(workspace, port), "the lake server never listened"
        assert _is_listening(workspace, port + 1), "the read-only endpoint never listened"
        run = workspace / ".mb" / "run"
        yield {
            "MB_LAKE": f"quack:localhost:{port}",
            "MB_LAKE_TOKEN": (run / "lake.token").read_text(encoding="utf-8").strip(),
            "MB_LAKE_READ": f"quack:localhost:{port + 1}",
            "MB_LAKE_READ_TOKEN": (run / "lake.read.token").read_text(encoding="utf-8").strip(),
        }


def test_a_fresh_lake_is_whole_and_compacts(mb, workspace: Path) -> None:
    assert mb("lake", "check").code == 0
    assert Lake.at(workspace).query("SELECT count(*) FROM lake.runs") == [(0,)]
    compacted = mb("lake", "compact")
    assert compacted.code == 0 and "compacted" in compacted.out


def test_a_catalog_at_an_older_spec_names_the_upgrade_that_reaches_every_project(
    mb, lakes: Sequence[Lake]
) -> None:
    _downgraded(lakes[0])
    older = mb("lake", "check", "--json")
    [finding] = json.loads(older.out)
    assert older.code == 1 and finding["kind"] == "catalog"
    assert "`mb lake upgrade`" in finding["detail"]
    assert "AUTOMATIC_MIGRATION" not in finding["detail"]
    upgraded = mb("lake", "upgrade", "--json")
    assert upgraded.code == 0, upgraded.said
    assert {row["lake"] for row in json.loads(upgraded.out)} == {str(it.catalog) for it in lakes}
    assert mb("lake", "check").code == 0


def test_a_second_workspace_reads_and_writes_the_served_lake(mb, elsewhere: Path, served) -> None:
    ran = mb("query", "SELECT count(*) AS n FROM lake.runs", "--json", cwd=elsewhere, env=served)
    assert ran.code == 0, ran.said
    assert json.loads(ran.out) == [{"n": 0}]
    done = subprocess.run(
        [sys.executable, "-c", _WRITE_FROM_FAR],
        cwd=elsewhere,
        env={**os.environ, **served},
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.splitlines() == ["[(1,)]", "[(1,)]"]
    assert not (elsewhere / ".mb").exists(), "a served lake keeps nothing on the client"
    # The server's own workspace sees the row through its files.
    ran = mb(
        "query", "SELECT count(*) AS n FROM lake.schema_log WHERE spec = 'from-far'", "--json"
    )
    assert json.loads(ran.out) == [{"n": 1}]


def test_a_blocking_served_query_obeys_the_server_timeout(mb, workspace: Path, served) -> None:
    """Only the server's own 10 s cap can end the aggregate, as client limits are not forwarded."""
    row = {"ts": datetime.now(UTC), "version": 0, "spec": "timeout", "engine": "it"}
    Lake.at(workspace).append(schema.schema_log, [row] * 3000)
    cross = "lake.schema_log a, lake.schema_log b, lake.schema_log c, lake.schema_log d"
    ran = mb("query", f"SELECT count(*) FROM {cross}", "--timeout", "0.02", env=served, timeout=20)
    assert ran.code != 0 and "timed out" in ran.err
    assert mb("query", "SELECT count(*) FROM lake.schema_log", env=served).code == 0


def test_a_served_lake_refuses_maintenance_and_strangers(mb, elsewhere: Path, served) -> None:
    compact = mb("lake", "compact", cwd=elsewhere, env=served)
    assert compact.code == 1 and "served from" in compact.err
    stranger = {**served, "MB_LAKE_READ_TOKEN": "not-the-token"}
    ran = mb("query", "SELECT 1 FROM lake.runs", cwd=elsewhere, env=stranger)
    assert ran.code == 1 and "Authentication failed" in ran.err


def test_a_session_lets_go_of_its_connection_when_closed_or_collected(lake: Lake) -> None:
    """The exit-time finalizer closes what `close` does, and a kept closed connection leaks."""
    session = lake.session()
    closed = weakref.ref(session.run(lambda connection: connection))
    session.close()
    gc.collect()
    assert closed() is None
    assert session.rows("SELECT 2") == [(2,)]
    attached = weakref.ref(session.run(lambda connection: connection))
    del session
    gc.collect()
    assert attached() is None
