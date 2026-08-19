import os
from typing import TYPE_CHECKING, NoReturn

import pytest
from plumbum import local

from mainboard import Board, Fleet, HostFacts, Job, MissionError, Survey
from mainboard.board import ProviderJob
from mainboard.dispatch import Handle, HostSetup, Verdict
from mainboard.dispatch.backends import (
    Account,
    Delivery,
    LogSource,
    ProviderBackend,
    Standing,
)
from mainboard.dispatch.schedulers import JobState
from mainboard.dispatch.state import RunRecord

from .dispatch.backends.conftest import BareBackend

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

_MIYABI_G = "miyabi-g"


class Replaced(Exception):
    """What the injected exec seam raises, standing in for this process being replaced."""


@pytest.fixture
def board(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> Board:
    """A board over the fixture manifest, its dispatch cache isolated inside the workspace."""
    monkeypatch.chdir(workspace)
    return Board(workspace)


@pytest.fixture
def provisioned(board: Board, tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    """A finished `default` installation and a stub `pixi` on PATH, yielding the binary's path.

    pixi stamps the fingerprint only once an install completes, so writing it is what makes
    the workspace look provisioned, and the stub is what a shell would have been replaced by.
    """
    fingerprint = board.root / ".mainboard" / ".pixi" / "envs" / "default" / "conda-meta"
    fingerprint.mkdir(parents=True)
    (fingerprint / ".pixi-environment-fingerprint").write_text("installed\n")
    bindir = tmp_path_factory.mktemp("bin")
    binary = bindir / "pixi"
    binary.write_text("#!/bin/sh\n")
    binary.chmod(0o755)
    with local.env(PATH=f"{bindir}{os.pathsep}{local.env['PATH']}"):
        yield str(binary)


def test_on_binds_the_host_and_shares_the_loaded_manifest(board: Board) -> None:
    assert board.local and board.host == "local"
    bound = board.on("miyabi-g")
    assert bound.host == "miyabi-g" and not bound.local
    assert bound.manifest is board.manifest
    assert bound.dispatcher is board.dispatcher
    assert bound.plan().containerized


def test_local_run_executes_through_the_wrapped_line(board: Board) -> None:
    assert board.run("true") == 0
    assert board.run("false") == 1


def test_run_hands_a_declared_task_to_pixi_and_anything_else_to_the_shell(
    board: Board, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`test` is a task in the fixture manifest, so the shell never sees that word at all."""
    staged: list[str] = []

    def capture(plan, root, *, command, containerize=None):
        staged.append(command)
        return "true"

    monkeypatch.setattr("mainboard.board.wrap", capture)

    board.run("test --quiet")
    board.run("pytest --quiet")

    assert staged == [
        "pixi run --manifest-path .mainboard/pixi.toml --frozen -e default test --quiet",
        "pytest --quiet",
    ]


def test_remote_root_comes_from_the_profile_or_refuses(board: Board) -> None:
    assert board.on("miyabi-g").remote_root() == "/work/xg25g007/x10537/projects"
    with pytest.raises(MissionError, match=r"set \[hosts.gold\] root"):
        board.on("gold").remote_root()


def test_local_facts_are_the_wire_snapshot(board: Board) -> None:
    facts = board.facts()
    assert isinstance(facts, HostFacts)
    assert facts.schema_version >= 1


def test_submit_resolves_profile_defaults_and_expressions(
    board: Board, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, object] = {}

    def fake_run(plan, cmd, *, root, resources, containerize=None, **extra):
        seen.update(
            queue=resources.queue,
            walltime=resources.walltime,
            mem=resources.mem_gb,
            account=resources.account,
            containerized=containerize is not None,
            root=root,
        )
        return Handle(id="77", host=plan.host, root=root, kind=plan.profile.kind)

    monkeypatch.setattr(board.dispatcher, "run", fake_run)
    job = board.on("miyabi-g").submit("python -m exp.run", attempt=2)
    assert isinstance(job, Job) and job.handle.id == "77"
    assert seen == {
        "queue": "debug-g",
        "walltime": "00:30:00",
        "mem": 100,
        "account": "xg25g007",
        "containerized": True,
        "root": "/work/xg25g007/x10537/projects",
    }


def test_submit_overrides_beat_the_defaults(board: Board, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    def fake_run(plan, cmd, *, root, resources, **extra):
        seen.update(queue=resources.queue, mem=resources.mem_gb)
        return Handle(id="78", host=plan.host, root=root, kind=plan.profile.kind)

    monkeypatch.setattr(board.dispatcher, "run", fake_run)
    board.on("miyabi-g").submit("true", queue="short-g", mem_gb=64, attempt=1)
    assert seen == {"queue": "short-g", "mem": 64}


def test_job_wait_and_pull_delegate_to_the_dispatcher(
    board: Board, monkeypatch: pytest.MonkeyPatch
) -> None:
    handle = Handle(id="9", host="miyabi-g", root="/work/p", kind="pbs")
    verdict = Verdict(verdict="ok")
    monkeypatch.setattr(
        board.dispatcher, "await_many", lambda handles, **kw: {handles[0]: verdict}
    )
    pulled: list[Handle] = []
    monkeypatch.setattr(board.dispatcher, "fetch", lambda h, **kw: pulled.append(h))
    job = Job(board.on("miyabi-g"), handle)
    assert job.wait() is verdict
    job.pull()
    assert pulled == [handle]


class FakeProvisioner:
    """A `Provisioner` double recording what a local install compiled and activated."""

    calls: list[tuple[str, object]] = []

    def __init__(self, root, manifest):
        FakeProvisioner.calls.append(("init", root))

    def activate(self, env, *, modules):
        FakeProvisioner.calls.append(("activate", (env, dict(modules))))
        return f"/repo/.mainboard/{env}-activate.sh"

    def provision(self, env, *, resolve):
        FakeProvisioner.calls.append(("provision", (env, resolve)))


def test_install_provisions_and_activates_locally(
    board: Board, monkeypatch: pytest.MonkeyPatch
) -> None:
    FakeProvisioner.calls = []
    monkeypatch.setattr("mainboard.board.Provisioner", FakeProvisioner)
    setup = board.install("serving", resolve=True)
    assert ("provision", ("serving", True)) in FakeProvisioner.calls
    assert ("activate", ("serving", {})) in FakeProvisioner.calls
    assert setup.host == "local"
    assert setup.installer == "in-place"
    assert setup.activate.endswith("serving-activate.sh")
    assert setup.tool


def test_install_takes_its_modules_from_a_named_profile(
    board: Board, monkeypatch: pytest.MonkeyPatch
) -> None:
    FakeProvisioner.calls = []
    monkeypatch.setattr("mainboard.board.Provisioner", FakeProvisioner)
    board.install(profile=_MIYABI_G)
    assert ("activate", ("default", {"singularity": "4.2.1"})) in FakeProvisioner.calls


def test_install_on_a_host_runs_the_onboarding(
    board: Board, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, object] = {}
    report = HostSetup(host="gold", root="/repo", installer="uv")

    class FakeOnboarding:
        def __init__(self, dispatcher, plan, *, root, artifact, resolve, watch):
            seen.update(
                host=plan.host,
                root=root,
                env=plan.env,
                artifact=tuple(artifact),
                resolve=resolve,
                containerized=plan.containerized,
            )

        def run(self):
            return report

    monkeypatch.setattr("mainboard.board.Onboarding", FakeOnboarding)
    assert board.on(_MIYABI_G).install("serving") is report
    assert seen == {
        "host": _MIYABI_G,
        "root": "/work/xg25g007/x10537/projects",
        "env": "serving",
        "artifact": (
            ".mainboard/pixi.toml",
            ".mainboard/pixi.lock",
            ".mainboard/state.toml",
        ),
        "resolve": False,
        "containerized": False,
    }


def test_installing_a_host_provisions_the_environment_its_profile_names(
    board: Board, monkeypatch: pytest.MonkeyPatch
) -> None:
    """gold declares `env = "serving"`, so setting gold up must not fall back to default."""
    seen: dict[str, object] = {}

    class FakeOnboarding:
        def __init__(self, dispatcher, plan, *, root, artifact, resolve, watch):
            seen.update(env=plan.env)

        def run(self):
            return HostSetup(host="gold", root="/repo")

    monkeypatch.setattr("mainboard.board.Onboarding", FakeOnboarding)
    board.on("gold").install()
    assert seen == {"env": "serving"}


def test_shell_replaces_this_process_with_a_frozen_pixi_shell(
    board: Board, provisioned: str
) -> None:
    """The terminal goes to `pixi shell`, pinned to the workspace and forbidden to solve."""
    seen: list[tuple[str, list[str]]] = []

    def replace(path: str, argv: list[str]) -> NoReturn:
        seen.append((path, argv))
        raise Replaced

    with pytest.raises(Replaced):
        board.shell(replace=replace)

    manifest = str(board.root / ".mainboard" / "pixi.toml")
    argv = [provisioned, "shell", "--manifest-path", manifest, "--frozen", "-e", "default"]
    assert seen == [(provisioned, argv)]


def test_shell_refuses_an_environment_nothing_provisioned(board: Board) -> None:
    """A shell on the system interpreter is the failure the refusal exists to prevent."""
    with pytest.raises(MissionError, match="Run `mainboard install serving`"):
        board.shell("serving")


def test_shell_refuses_on_a_bound_host(board: Board) -> None:
    with pytest.raises(MissionError, match="this machine only"):
        board.on("gold").shell()


def test_job_logs_and_kill_use_keyword_scheduler_calls(
    board: Board, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, str]] = []

    class FakeScheduler:
        def cancel(self, remote, root, *, handle):
            calls.append(("cancel", handle))

        def logs(self, remote, root, *, handle):
            calls.append(("logs", handle))
            return "the log"

    class FakeConnection:
        def __enter__(self):
            return "remote"

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr("mainboard.board.pick", lambda profile: FakeScheduler())
    monkeypatch.setattr("mainboard.board.connection", lambda host: FakeConnection())
    job = Job(board.on("miyabi-g"), Handle(id="9", host="miyabi-g", root="/work/p", kind="pbs"))
    assert job.logs() == "the log"
    job.kill()
    assert calls == [("logs", "9"), ("cancel", "9")]


def test_remote_facts_parse_the_last_json_line(
    board: Board, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = HostFacts(schema_version=1, hostname="fake-remote").model_dump_json()

    class FakeCommand:
        def __call__(self):
            return f"module chatter\n{payload}\n"

    class FakeRemote:
        def __getitem__(self, name):
            return FakeBound()

    class FakeBound:
        def __getitem__(self, argv):
            return FakeCommand()

    class FakeConnection:
        def __enter__(self):
            return FakeRemote()

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr("mainboard.board.connection", lambda host: FakeConnection())
    facts = board.on("miyabi-g").facts()
    assert facts.hostname == "fake-remote"


def test_remote_run_streams_over_the_connection(
    board: Board, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeRemote:
        def __getitem__(self, name):
            return FakeBound()

    class FakeBound:
        def __getitem__(self, argv):
            return "bound"

    class FakeConnection:
        def __enter__(self):
            return FakeRemote()

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr("mainboard.board.connection", lambda host: FakeConnection())
    monkeypatch.setattr("mainboard.board._streamed", lambda command: 7)
    assert board.on("miyabi-g").run("true", container="none") == 7


def test_fleet_accessor_binds_this_board(board: Board) -> None:

    fleet = board.on("gold").fleet()
    assert isinstance(fleet, Fleet)


def test_compute_surveys_the_whole_workspace_whatever_host_the_board_is_bound_to(
    board: Board,
) -> None:
    survey = board.on("gold").compute()
    assert isinstance(survey, Survey)
    assert survey.board.manifest is board.manifest


class _FakeCloud(ProviderBackend, Account, Delivery, LogSource):
    """A registered `fakecloud`-kind backend carrying every capability, so nothing is refused.

    Module-level so it registers exactly once.
    """

    name = "fakecloud"
    submitted: list[str] = []
    cancelled: list[str] = []

    def cancel(self, handle):
        _FakeCloud.cancelled.append(handle)

    def deliver(self, handle, *, path):
        return None

    def logs(self, handle):
        return "cloud log"

    def standing(self):
        return Standing(keyed=True, note="a fake cloud is always paid up")

    def state(self, handle):
        return JobState(handle=handle, state="finished", exit_code=0, verdict="ok")

    def submit(self, plan, command, resources):
        _FakeCloud.submitted.append(command)
        return "cloud-1"


def _submit_to_fakecloud(
    board: Board, monkeypatch: pytest.MonkeyPatch, *, fetch: str | None = None
) -> ProviderJob:
    """A submitted `ProviderJob` routed to the shared `fakecloud`-kind backend."""
    _FakeCloud.submitted = []
    _FakeCloud.cancelled = []
    manifest = board.manifest.model_copy(
        update={
            "hosts": {
                **board.manifest.hosts,
                "cloudbox": board.manifest.profile("gold").model_copy(
                    update={"kind": "fakecloud"}
                ),
            }
        }
    )
    monkeypatch.setitem(board.shared, "manifest", manifest)
    monkeypatch.setitem(board.shared, "resolver", None)
    return board.on("cloudbox").submit("python train.py", mem_gb=8, fetch=fetch)


def test_submit_routes_provider_kinds_to_the_backend(
    board: Board, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = _submit_to_fakecloud(board, monkeypatch)
    assert isinstance(job, ProviderJob)
    assert job.backend.submitted == ["python train.py"]


def test_provider_job_wait_logs_kill_and_pull_delegate_to_the_backend(
    board: Board, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = _submit_to_fakecloud(board, monkeypatch, fetch="results/")
    states = iter(["running", "ok"])

    def flipping(handle):
        return JobState(handle=handle, state="x", verdict=next(states))

    job.backend.state = flipping
    verdict = job.wait(poll=lambda seconds: None)
    assert verdict.ok and job.logs() == "cloud log"
    job.kill()
    job.pull()


def test_a_provider_submit_is_recorded_in_the_same_cache_a_queued_job_lands_in(
    board: Board, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without the record no later process could settle the run, and the rental would bill on."""
    job = _submit_to_fakecloud(board, monkeypatch, fetch="results/run")
    record = board.dispatcher.cache.run(job.handle.id)
    assert (record.target, record.kind) == ("cloudbox", "fakecloud")
    assert (record.script, record.fetch_path) == ("python train.py", "results/run")
    assert record.verdict is None


def test_job_rebuilds_a_provider_run_as_the_kind_the_cache_recorded(
    board: Board, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fresh process addresses a rental the way the process that rented it did."""
    submitted = _submit_to_fakecloud(board, monkeypatch, fetch="results/run")
    rebuilt = board.job(submitted.handle.id)
    assert isinstance(rebuilt, ProviderJob)
    assert rebuilt.handle == submitted.handle
    assert rebuilt.backend.name == "fakecloud"


def test_a_provider_job_that_finishes_a_wait_has_its_rental_ended(
    board: Board, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A caller that blocked on the run is the last thing between it and an idle meter."""
    job = _submit_to_fakecloud(board, monkeypatch)
    assert job.wait(poll=lambda seconds: None).ok
    assert job.backend.cancelled == [job.handle.id]


def test_a_provider_job_refuses_a_capability_its_backend_never_had(board: Board) -> None:
    """The absence is discovered before the call, and answered with the backend's own advice."""
    handle = Handle(id="bare-1", host="cloudbox", root="", kind="bare", fetch_path="results/")
    job = ProviderJob(BareBackend(), handle)
    job.kill()
    assert job.wait(poll=lambda seconds: None).ok
    with pytest.raises(MissionError, match=r"bare backend keeps no logs; read bare-1\.log"):
        job.logs()
    with pytest.raises(MissionError, match="does not implement Delivery"):
        job.pull()


def test_a_provider_job_with_no_recorded_results_path_has_nothing_to_pull(board: Board) -> None:
    job = ProviderJob(BareBackend(), Handle(id="bare-1", host="cloudbox", root="", kind="bare"))
    with pytest.raises(LookupError, match="no fetch path"):
        job.pull()


def test_job_state_is_a_non_blocking_probe(board: Board, monkeypatch: pytest.MonkeyPatch) -> None:
    handle = Handle(id="9", host=_MIYABI_G, root="/work/p", kind="pbs")
    probed = JobState(handle="9", state="R", verdict="running")
    monkeypatch.setattr(
        board.dispatcher, "probe", lambda asked: probed if asked == handle else None
    )
    assert Job(board.on(_MIYABI_G), handle).state() is probed


def test_job_rebuilds_a_dispatched_run_from_the_cache(
    board: Board, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = RunRecord(
        handle="4242",
        target=_MIYABI_G,
        kind="pbs",
        script="job.sh",
        args="",
        git_sha="abc1234",
        dirty=0,
        submitted_at="2026-08-17T00:00:00",
        fetch_path="results/run",
    )
    monkeypatch.setattr(board.dispatcher.cache, "run", lambda handle, target=None: record)
    job = board.job(4242)
    assert job.handle.id == "4242"
    assert job.handle.host == _MIYABI_G
    assert job.handle.root == "/work/xg25g007/x10537/projects"
    assert job.handle.fetch_path == "results/run"
    assert job.board.host == _MIYABI_G
