"""Run the existing CLI scenarios inside the bounded M00 worker."""

import contextlib
import importlib.util
import io
import json
import os
import subprocess
import sys
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path

import pytest
from hypothesis import settings
from patos import FrozenModel

import mainboard
from mainboard import MissionError, logger
from mainboard.cli import build

from .constants import MB, SCRIPTS
from .legacy import LegacyRuntime

# Every property runs without Hypothesis's example database (a sandbox cannot write one) and
# without a per-example deadline (a loaded machine would fail a correct property).
settings.register_profile("mb", database=None, deadline=None)
settings.load_profile("mb")

_OUTCOMES: dict[str, dict[str, str]] = {}
_CATALOG_TEST = "test_offers_list_every_cloud_and_mark_what_mb_rents"


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    previous = _OUTCOMES.get(report.nodeid, {})
    if previous.get("outcome") == "failed":
        return
    if report.failed or report.skipped or report.when == "call":
        _OUTCOMES[report.nodeid] = {
            "outcome": "failed" if hasattr(report, "wasxfail") else report.outcome,
            "phase": report.when or "",
            "reason": str(report.longrepr) if report.failed or report.skipped else "",
        }


# Explicit infrastructure obligations, never xfails standing in for bug regressions.
_SOLVER = {
    "test_install_solves_a_stale_lock_and_locked_refuses_it": "M00-SOLVER/M15",
    "test_as_is_runs_what_is_installed_without_solving": "M00-SOLVER/M15",
    "test_a_failed_solve_leaves_the_manifest_and_lock_as_they_were": "M00-SOLVER/M14",
    "test_upgrade_leaves_a_member_named_by_version": "M00-SOLVER/M16",
}

# A skip reason starting `M00-` is an explicit infrastructure obligation; so are the two a fix
# mirror cannot avoid.
_LANE_SKIP = "M00-"
_MIRROR_SKIPS = (
    "the legacy archive needs the git history of this tree",
    "not inside the monorepo",
)


def _unexpected_skips() -> list[str]:
    """The skipped tests that neither the baseline nor this lane's obligations account for."""
    baseline = json.loads(
        (Path(__file__).parent / "gates/baseline.json").read_text(encoding="utf-8")
    )
    known = {
        case["nodeid"]
        for case in baseline["cases"]
        if case["classification"] in {"skipped", "excluded"}
    }
    accounted = (_LANE_SKIP, *(_MIRROR_SKIPS if os.environ.get("MB_FIX_MIRROR") else ()))
    return [
        node
        for node, report in _OUTCOMES.items()
        if report["outcome"] == "skipped"
        and node not in known
        and not any(reason in report["reason"] for reason in accounted)
    ]


def pytest_sessionfinish(session: pytest.Session) -> None:
    if session.config.option.collectonly or not os.environ.get("MB_GATE_REPORT"):
        return
    unexpected = _unexpected_skips()
    if unexpected or any(report["outcome"] == "failed" for report in _OUTCOMES.values()):
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
    (Path(os.environ["MB_GATE_REPORT"]) / "outcomes.json").write_text(
        json.dumps(
            {
                "cases": _OUTCOMES,
                "unexpected_skips": unexpected,
                "not_run": sorted({item.nodeid for item in session.items} - _OUTCOMES.keys()),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def pytest_sessionstart(session: pytest.Session) -> None:
    tree = Path(__file__).resolve().parents[1]
    source = str(tree / "src")
    if Path(mainboard.__file__).resolve() != tree / "src/mainboard/__init__.py":
        raise pytest.UsageError(f"wrong mainboard import: {mainboard.__file__}")
    # Every `mb` a test starts imports this tree, whatever the interpreter's install holds.
    others = [path for path in os.environ.get("PYTHONPATH", "").split(os.pathsep) if path]
    os.environ["PYTHONPATH"] = os.pathsep.join([source, *(p for p in others if p != source)])
    root = Path(os.environ.get("MB_GATE_ROOT", "/nonexistent-M00-root"))
    if os.environ.get("MB_GATE_INSPECT") != "1" and not (root / "os-isolation-proved").is_file():
        raise pytest.UsageError(
            "M00-OS: use sh integration/gates/run.sh suite; fallback forbidden"
        )


def _skipped(item: pytest.Item) -> str:
    """Why this lane never runs `item`, empty when it runs; the first reason is reported."""
    if sys.platform != "darwin" and item.get_closest_marker("swift_native"):
        return "M00-SWIFT27: native SwiftSyntax cases run in the macOS/Xcode 27 lane"
    if reason := _SOLVER.get(item.name):
        return f"{reason}: needs an offline native fixture"
    if item.path.name == "test_remote.py":
        return "M00-FLEET/M17: real fleet always forbidden"
    served = isinstance(item, pytest.Function) and "served" in item.fixturenames
    return "M00-QUACK/M22: isolated Linux TCP lane" if sys.platform != "linux" and served else ""


def pytest_collection_modifyitems(items: Sequence[pytest.Item]) -> None:
    for item in items:
        if reason := _skipped(item):
            item.add_marker(pytest.mark.skip(reason=reason))


@pytest.fixture(autouse=True)
def _own_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Start each test in its own empty directory: the checkout that ran it names no workspace."""
    monkeypatch.chdir(tmp_path)


def _pg(name: str, *args: str) -> subprocess.CompletedProcess[str]:
    """Run the environment's PostgreSQL program `name`, never a system one."""
    return subprocess.run(
        [str(SCRIPTS / name), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
        check=True,
    )


def _stop(data: Path) -> None:
    if (data / "postmaster.pid").exists():
        _pg("pg_ctl", "-D", str(data), "-m", "immediate", "-w", "-t", "20", "stop")


@pytest.fixture
def postgres(tmp_path: Path, request: pytest.FixtureRequest) -> str:
    """A new catalog on a private Unix socket; never use a system PostgreSQL or TCP port."""
    missing = [
        name for name in ("postgres", "initdb", "pg_ctl", "psql") if not (SCRIPTS / name).is_file()
    ]
    if missing:
        pytest.skip(f"M00-POSTGRES: conda-forge postgresql missing from environment: {missing}")
    data, sockets = tmp_path / "pgdata", tmp_path / "socket"
    sockets.mkdir(mode=0o700)
    _pg("initdb", "-D", str(data), "-A", "trust", "-U", "gate", "--no-locale")
    # unix_socket_permissions keeps even temporary trust auth private to this user.
    escaped = str(sockets).replace("'", "''")
    with (data / "postgresql.conf").open("a", encoding="utf-8") as config:
        config.write(
            f"\nlisten_addresses = ''\nunix_socket_directories = '{escaped}'\n"
            "unix_socket_permissions = 0700\n"
        )
    request.addfinalizer(lambda: _stop(data))
    _pg("pg_ctl", "-D", str(data), "-l", str(tmp_path / "postgres.log"), "-w", "-t", "20", "start")
    return f"host={sockets} dbname=postgres user=gate"


class Ran(FrozenModel):
    """One finished `mb` process."""

    code: int
    out: str
    err: str

    @property
    def said(self) -> str:
        return self.out + self.err


@pytest.fixture
def legacy_runtime(tmp_path: Path) -> LegacyRuntime:
    """The unmodified c97cb81 archive, extracted under `tmp_path` with its own interpreter."""
    return LegacyRuntime.extract(tmp_path)


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """A fresh workspace holding only a manifest."""
    (tmp_path / "mb.toml").write_text('[workspace]\nname = "it"\n', encoding="utf-8")
    return tmp_path


def _console(workspace: Path, extra: dict[str, str]) -> Callable[..., Ran]:
    """`mb` as a process, refusing any traceback in what it printed."""

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
            env={**os.environ, "NO_COLOR": "1", **extra, **(env or {})},
            check=False,
        )
        ran = Ran(code=done.returncode, out=done.stdout, err=done.stderr)
        assert "Traceback (most recent call last)" not in ran.said, ran.said
        return ran

    return run


def in_process(workspace: Path) -> Callable[..., Ran]:
    """`mb` called here, for the inspect lane where no child process may start."""

    def run(
        *args: str,
        cwd: Path | None = None,
        timeout: float = 120,
        env: dict[str, str] | None = None,
    ) -> Ran:
        assert env is None, "the inspect lane cannot set a child's environment"
        out, err = io.StringIO(), io.StringIO()
        with (
            contextlib.chdir(cwd or workspace),
            contextlib.redirect_stdout(out),
            contextlib.redirect_stderr(err),
        ):
            try:
                code = build()(args, exit_on_error=False)
            except SystemExit as status:
                code = status.code
            except MissionError as fault:
                logger.error(str(fault))
                code = 1
        return Ran(code=code or 0, out=out.getvalue(), err=err.getvalue())

    return run


@pytest.fixture
def mb(workspace: Path, request: pytest.FixtureRequest) -> Iterator[Callable[..., Ran]]:
    """Run `mb` with `args` inside the workspace."""
    if os.environ.get("MB_GATE_INSPECT") == "1":
        yield in_process(workspace)
        return
    if MB is None:
        pytest.skip("the mb console script is not installed beside this interpreter")
    catalog = request.node.name == _CATALOG_TEST
    if catalog and importlib.util.find_spec("catalog_double") is None:
        pytest.skip("M00-CATALOG: the recorded offer exists only in a gate worker (run.sh)")
    calls = workspace / "catalog-calls.jsonl"
    yield _console(workspace, {"MB_GATE_CATALOG": str(calls)} if catalog else {})
    if catalog:
        assert [json.loads(line) for line in calls.read_text(encoding="utf-8").splitlines()] == [
            {"gpu_name": ["RTX4090"], "min_gpu_count": 1, "max_gpu_count": 1}
        ]


def git(where: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Run git in `where`, local-path submodules allowed and an author named."""
    settings = ["protocol.file.allow=always", "user.name=it", "user.email=it@example.invalid"]
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
    """The workspace as a repository whose manifest is committed and pushed to a bare `origin`.

    That remote makes it the workspace's own, and the repository has an author.
    """
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
