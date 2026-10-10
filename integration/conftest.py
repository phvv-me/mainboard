"""Run the existing CLI scenarios inside the bounded M00 worker."""

import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import sysconfig
import tarfile
import venv
from collections.abc import Iterator
from pathlib import Path

import pytest
from patos import FrozenModel

import mainboard

# The console script installed beside the interpreter running the tests.
MB = shutil.which("mb", path=sysconfig.get_path("scripts"))
_OUTCOMES: dict[str, dict[str, str]] = {}


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    previous = _OUTCOMES.get(report.nodeid, {})
    if previous.get("outcome") == "failed":
        return
    if report.failed or report.skipped or report.when == "call":
        _OUTCOMES[report.nodeid] = {
            "outcome": "failed" if hasattr(report, "wasxfail") else report.outcome,
            "phase": report.when,
            "reason": str(report.longrepr) if report.failed or report.skipped else "",
        }


def pytest_sessionfinish(session: pytest.Session) -> None:
    if session.config.option.collectonly or not os.environ.get("MB_GATE_REPORT"):
        return
    baseline = json.loads(
        (Path(__file__).parent / "gates/baseline.json").read_text(encoding="utf-8")
    )
    known_skips = {case["nodeid"] for case in baseline["cases"]
                   if case["classification"] in {"skipped", "excluded"}}
    known_skips.update({
        "integration/test_lake.py::test_a_second_workspace_reads_and_writes_the_served_lake",
        "integration/test_lake.py::test_a_served_lake_refuses_maintenance_and_strangers",
        "integration/gates/test_harness.py::test_postgres_is_a_temporary_catalog",
    })
    unexpected = [node for node, report in _OUTCOMES.items()
                  if report["outcome"] == "skipped" and node not in known_skips]
    failed = any(report["outcome"] == "failed" for report in _OUTCOMES.values())
    if unexpected or failed:
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
    destination = Path(os.environ["MB_GATE_REPORT"]) / "outcomes.json"
    destination.write_text(json.dumps({
        "cases": _OUTCOMES, "unexpected_skips": unexpected,
        "not_run": sorted({item.nodeid for item in session.items} - _OUTCOMES.keys()),
    }, indent=2) + "\n", encoding="utf-8")


def pytest_sessionstart(session: pytest.Session) -> None:
    tree = Path(__file__).resolve().parents[1]
    if Path(mainboard.__file__).resolve() != tree / "src/mainboard/__init__.py":
        raise pytest.UsageError(f"wrong mainboard import: {mainboard.__file__}")
    if session.config.option.collectonly and os.environ.get("MB_GATE_INSPECT") == "1":
        return
    root = Path(os.environ.get("MB_GATE_ROOT", "/nonexistent-M00-root"))
    if not (root / "os-isolation-proved").is_file():
        raise pytest.UsageError(
            "M00-OS: use sh integration/gates/run.sh suite; fallback forbidden"
        )


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    # Explicit infrastructure obligations, never xfails standing in for bug regressions.
    external = {
        "test_install_solves_a_stale_lock_and_locked_refuses_it": "M00-SOLVER/M15",
        "test_as_is_runs_what_is_installed_without_solving": "M00-SOLVER/M15",
        "test_a_failed_solve_leaves_the_manifest_and_lock_as_they_were": "M00-SOLVER/M14",
        "test_upgrade_leaves_a_member_named_by_version": "M00-SOLVER/M16",
    }
    for item in items:
        if reason := external.get(item.name):
            item.add_marker(pytest.mark.skip(reason=f"{reason}: needs an offline native fixture"))
        if item.path.name == "test_remote.py":
            item.add_marker(pytest.mark.skip(reason="M00-FLEET/M17: real fleet always forbidden"))
        if sys.platform != "linux" and "served" in item.fixturenames:
            item.add_marker(pytest.mark.skip(reason="M00-QUACK/M22: isolated Linux TCP lane"))


@pytest.fixture
def postgres(tmp_path: Path) -> Iterator[str]:
    """A new catalog on a private Unix socket; never use a system PostgreSQL or TCP port."""
    binaries = Path(sysconfig.get_path("scripts"))
    missing = [name for name in ("postgres", "initdb", "pg_ctl", "psql")
               if not (binaries / name).is_file()]
    if missing:
        pytest.skip(f"M00-POSTGRES: conda-forge postgresql missing from environment: {missing}")
    data, sockets = tmp_path / "pgdata", tmp_path / "socket"
    sockets.mkdir(mode=0o700)

    def run(name: str, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [str(binaries / name), *args], capture_output=True, text=True, encoding="utf-8",
            timeout=30, check=True,
        )

    run("initdb", "-D", str(data), "-A", "trust", "-U", "gate", "--no-locale")
    # unix_socket_permissions keeps even temporary trust auth private to this user.
    with (data / "postgresql.conf").open("a", encoding="utf-8") as config:
        escaped = str(sockets).replace("'", "''")
        config.write(f"\nlisten_addresses = ''\nunix_socket_directories = '{escaped}'\n"
                     "unix_socket_permissions = 0700\n")
    try:
        run("pg_ctl", "-D", str(data), "-l", str(tmp_path / "postgres.log"), "-w", "-t", "20",
            "start")
        yield f"host={sockets} dbname=postgres user=gate"
    finally:
        if (data / "postmaster.pid").exists():
            run("pg_ctl", "-D", str(data), "-m", "immediate", "-w", "-t", "20", "stop")


class Ran(FrozenModel):
    """One finished `mb` process."""

    code: int
    out: str
    err: str

    @property
    def said(self) -> str:
        return self.out + self.err


class LegacyRuntime(FrozenModel):
    root: Path
    interpreter: Path
    archive_sha256: str

    def run(self, program: str) -> subprocess.CompletedProcess[str]:
        # Replacement, not concatenation: the new checkout must never supply legacy modules.
        environment = {**os.environ, "PYTHONPATH": str(self.root / "src")}
        return subprocess.run(
            [str(self.interpreter), "-c", program], cwd=self.root, env=environment,
            capture_output=True, text=True, encoding="utf-8", timeout=30, check=True,
        )


@pytest.fixture
def legacy_runtime(tmp_path: Path) -> LegacyRuntime:
    revision = "c97cb81daed05e8b9d0ed21f87a7427075487562"
    archive = subprocess.run(
        ["git", "-C", str(Path(__file__).resolve().parents[1]), "archive", revision],
        capture_output=True, timeout=30, check=True,
    ).stdout
    root = tmp_path / "legacy"
    root.mkdir()
    with tarfile.open(fileobj=io.BytesIO(archive)) as tree:
        tree.extractall(root, filter="data")
    prefix = tmp_path / "interpreter"
    venv.EnvBuilder(system_site_packages=True, with_pip=False).create(prefix)
    runtime = LegacyRuntime(
        root=root, interpreter=prefix / "bin/python",
        archive_sha256=hashlib.sha256(archive).hexdigest(),
    )
    proof = runtime.run("import mainboard; print(mainboard.__file__)")
    assert Path(proof.stdout.strip()).resolve() == root / "src/mainboard/__init__.py"
    return runtime


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """A fresh workspace holding only a manifest."""
    (tmp_path / "mb.toml").write_text('[workspace]\nname = "it"\n', encoding="utf-8")
    return tmp_path


@pytest.fixture
def mb(workspace: Path, request: pytest.FixtureRequest):
    """Run `mb` with `args` inside the workspace, refusing any traceback in what it printed."""
    if MB is None:
        pytest.skip("the mb console script is not installed beside this interpreter")
    catalog = request.node.name == "test_offers_list_every_cloud_and_mark_what_mb_rents"
    calls = workspace / "catalog-calls.jsonl"
    fake = {"MB_GATE_CATALOG": str(calls)} if catalog else {}

    def run(
        *args: str,
        cwd: Path | None = None,
        timeout: float = 120,
        env: dict[str, str] | None = None,
    ) -> Ran:
        done = subprocess.run(
            [MB, *args],
            cwd=cwd or workspace,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=min(timeout, 120),
            env={**os.environ, "NO_COLOR": "1", **fake, **(env or {})},
            check=False,
        )
        ran = Ran(code=done.returncode, out=done.stdout, err=done.stderr)
        assert "Traceback (most recent call last)" not in ran.said, ran.said
        return ran

    yield run
    if catalog:
        assert [json.loads(line) for line in calls.read_text(encoding="utf-8").splitlines()] == [
            {"gpu_name": ["rtx 4090"], "min_gpu_count": 1, "max_gpu_count": 1}
        ]


def git(where: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Run git in `where`, local-path submodules allowed and an author named."""
    settings = ("protocol.file.allow=always", "user.name=it", "user.email=it@example.invalid")
    flags = [word for setting in settings for word in ("-c", setting)]
    return subprocess.run(
        ["git", "-C", str(where), *flags, *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )


def committed(where: Path) -> Path:
    """A repository at `where` holding one commit."""
    where.mkdir(parents=True)
    git(where, "init", "--quiet", "--initial-branch=main")
    (where / "README").write_text("one\n", encoding="utf-8", newline="\n")
    git(where, "add", "README")
    git(where, "commit", "--quiet", "-m", "one")
    return where


@pytest.fixture
def tracked(workspace: Path, tmp_path_factory) -> Path:
    """The workspace as a repository with an author, its manifest committed and pushed to a bare
    `origin`, the remote that makes it the workspace's own."""
    remote = tmp_path_factory.mktemp("remotes") / "root.git"
    git(workspace, "init", "--quiet", "--bare", "--initial-branch=main", remote.as_posix())
    git(workspace, "init", "--quiet", "--initial-branch=main")
    git(workspace, "config", "user.name", "it")
    git(workspace, "config", "user.email", "it@example.invalid")
    git(workspace, "remote", "add", "origin", remote.as_posix())
    git(workspace, "add", "-A")
    git(workspace, "commit", "--quiet", "-m", "manifest")
    git(workspace, "push", "--quiet", "-u", "origin", "main")
    return workspace
