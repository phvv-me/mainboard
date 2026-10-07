"""Real POSIX file limits, fragmented evidence, and atomic recovery after exhaustion."""

import errno
import hashlib
import json
import os
import resource
import shlex
import sys
from collections.abc import Generator
from contextlib import closing, contextmanager
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import psutil
import pytest

from mainboard.runtime.job import Job, WorkspaceActivation
from mainboard.runtime.runner import Runner
from mainboard.state import Lake
from mainboard.state.blobs import Blobs

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX RLIMIT_NOFILE")


@contextmanager
def _file_limit(soft: int) -> Generator[None]:
    original = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (soft, original[1]))
    try:
        yield
    finally:
        resource.setrlimit(resource.RLIMIT_NOFILE, original)


def _limit() -> tuple[int, int]:
    return resource.getrlimit(resource.RLIMIT_NOFILE)


@pytest.fixture
def fragmented(workspace: Path) -> tuple[Lake, Blobs, set[str]]:
    lake = Lake.at(workspace).ready()
    blobs = Blobs(lake)
    digests = set()
    with lake.open(write=True) as connection:
        for index in range(128):
            payload = f"object-{index}".encode()
            digest = hashlib.sha256(payload).hexdigest()
            digests.add(digest)
            connection.execute("BEGIN")
            blobs.stage(connection, {digest: payload})
            connection.execute("COMMIT")
    return lake, blobs, digests


def test_job_children_inherit_permitted_limit_and_caller_is_restored(tmp_path, capfd) -> None:
    job = Job(
        command=shlex.join(
            [
                sys.executable,
                "-c",
                "import json,resource; "
                "print(json.dumps(resource.getrlimit(resource.RLIMIT_NOFILE)))",
            ]
        ),
        root=str(tmp_path),
        activation=WorkspaceActivation(script="", prefix=sys.prefix, refusal="missing prefix"),
    )
    with _file_limit(256):
        hard = _limit()[1]
        assert Runner(job, environ=os.environ).run() == 0
        assert _limit() == (256, hard)
    soft, inherited = json.loads(capfd.readouterr().out.splitlines()[-1])
    assert soft > 256 and inherited == hard


def test_fragmented_membership_recovers_and_releases_files(fragmented) -> None:
    lake, blobs, digests = fragmented
    session = lake.session()
    before = psutil.Process().num_fds()
    with _file_limit(before + 48):
        with (
            closing(duckdb.connect()) as connection,
            pytest.raises(duckdb.IOException, match="Too many open files"),
        ):
            lake.attach(connection)
            blobs.held(connection, digests)
        assert session.run(lambda connection: blobs.held(connection, digests)) == digests
        assert session.run(lambda connection: blobs.held(connection, digests)) == digests
        session.close()
        assert psutil.Process().num_fds() <= before + 4


def test_failed_transaction_rolls_back_and_can_be_retried(fragmented) -> None:
    lake, blobs, digests = fragmented
    attempts = 0

    def work(connection: duckdb.DuckDBPyConnection) -> set[str]:
        nonlocal attempts
        attempts += 1
        connection.execute(
            "INSERT INTO lake.schema_log (ts,version,spec,engine) VALUES (?,0,'retry','test')",
            [datetime.now(UTC)],
        )
        if attempts == 1:
            raise OSError(errno.EMFILE, "hard file budget exhausted")
        return blobs.held(connection, digests)

    with _file_limit(psutil.Process().num_fds() + 48):
        with pytest.raises(OSError, match="hard file budget exhausted"):
            lake.transact(work)
        assert attempts == 1
        assert lake.transact(work) == digests
    assert lake.query("SELECT count(*) FROM lake.schema_log WHERE spec='retry'") == [(1,)]


def test_hard_exhaustion_and_other_io_errors_escape_without_retry(workspace) -> None:
    session = Lake.at(workspace).ready().session()
    attempts = []

    def exhausted(connection: duckdb.DuckDBPyConnection) -> None:
        attempts.append(connection.execute("SELECT current_setting('threads')").fetchone()[0])
        raise OSError(errno.EMFILE, "Too many open files")

    with pytest.raises(OSError, match="Too many open files"):
        session.run(exhausted)
    assert len(attempts) == 1

    def unrelated(connection: duckdb.DuckDBPyConnection) -> None:
        del connection
        raise OSError(errno.EIO, "damaged storage")

    with pytest.raises(OSError, match="damaged storage"):
        session.run(unrelated)
    session.close()


def test_overlapping_lakes_restore_the_limit_after_last_close(workspace, tmp_path) -> None:
    first = Lake.at(workspace).ready().open()
    second = Lake.at(tmp_path / "other").ready().open()
    with _file_limit(256):
        hard = _limit()[1]
        left = first.__enter__()
        right = second.__enter__()
        try:
            assert left.execute("SELECT 1").fetchall() == [(1,)]
            assert right.execute("SELECT 1").fetchall() == [(1,)]
        finally:
            first.__exit__(None, None, None)
            assert _limit()[0] > 256
            second.__exit__(None, None, None)
        assert _limit() == (256, hard)
