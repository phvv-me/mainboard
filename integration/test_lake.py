"""The state lake's own verbs, and one lake served over Quack to a second workspace."""

import json
import os
import socket
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from .conftest import MB


def test_a_fresh_lake_is_whole_and_compacts(mb) -> None:
    assert mb("lake", "check").code == 0
    ran = mb("lake", "compact")
    assert ran.code == 0 and "compacted" in ran.out


@pytest.fixture
def served(workspace: Path) -> Iterator[dict[str, str]]:
    """The workspace's lake served on a free port, and the environment reaching it."""
    with socket.socket() as probe:
        probe.bind(("localhost", 0))
        port = probe.getsockname()[1]
    server = subprocess.Popen(
        [MB, "lake", "serve", "--port", str(port)],
        cwd=workspace,
        env={**os.environ, "NO_COLOR": "1"},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        waited = subprocess.run(
            [MB, "proc", "wait", "--port", f"localhost:{port}", "--timeout", "90"],
            cwd=workspace,
            check=False,
        )
        assert waited.returncode == 0, "the lake server never listened"
        token = (workspace / ".mb" / "run" / "lake.token").read_text(encoding="utf-8").strip()
        yield {"MB_LAKE": f"quack:localhost:{port}", "MB_LAKE_TOKEN": token}
    finally:
        subprocess.run([MB, "proc", "kill", str(server.pid)], check=False)
        server.wait(timeout=60)


def test_a_second_workspace_reads_and_writes_the_served_lake(
    mb, workspace, served, tmp_path_factory
) -> None:
    elsewhere = tmp_path_factory.mktemp("elsewhere")
    (elsewhere / "mb.toml").write_text('[workspace]\nname = "far"\n', encoding="utf-8")
    ran = mb("query", "SELECT count(*) AS n FROM lake.runs", "--json", cwd=elsewhere, env=served)
    assert ran.code == 0, ran.said
    assert json.loads(ran.out) == [{"n": 0}]

    # An append, a built read and a parameterized one, through the session every writer uses.
    write = (
        "from datetime import UTC, datetime\n"
        "from pathlib import Path\n"
        "from sqlalchemy import func, select\n"
        "from mb.state import schema\n"
        "from mb.state.lake import Lake\n"
        "session = Lake.at(Path.cwd()).session()\n"
        "log = schema.schema_log\n"
        "session.append(log, [{'ts': datetime.now(UTC), 'version': 0, "
        "'spec': 'from-far', 'engine': 'it'}])\n"
        "print(session.rows(select(func.count()).where(log.c.spec == 'from-far')))\n"
        "sql = 'SELECT count(*) FROM lake.schema_log WHERE spec = ?'\n"
        "print(session.rows(sql, ['from-far']))\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", write],
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


def test_a_served_lake_refuses_maintenance_and_strangers(mb, served, tmp_path_factory) -> None:
    elsewhere = tmp_path_factory.mktemp("elsewhere")
    (elsewhere / "mb.toml").write_text('[workspace]\nname = "far"\n', encoding="utf-8")
    ran = mb("lake", "compact", cwd=elsewhere, env=served)
    assert ran.code == 1 and "served from" in ran.err
    stranger = {**served, "MB_LAKE_TOKEN": "not-the-token"}
    ran = mb("query", "SELECT 1 FROM lake.runs", cwd=elsewhere, env=stranger)
    assert ran.code == 1 and "Authentication failed" in ran.err
