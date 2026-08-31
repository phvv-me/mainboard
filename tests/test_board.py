import os
from collections.abc import Callable, Iterator, Mapping
from threading import Event, Thread
from time import sleep
from types import TracebackType
from typing import TYPE_CHECKING, NoReturn

import pytest
from plumbum import local

from mainboard import Board, ExecutionPlan, Fleet, HostFacts, Job, MissionError, Survey
from mainboard.batch import Topic
from mainboard.board import ProviderJob
from mainboard.deps import Dependencies
from mainboard.dispatch import Handle, HostSetup, Verdict
from mainboard.dispatch.backends import Account, Delivery, LogSource, ProviderBackend, Standing
from mainboard.dispatch.schedulers import registry
from mainboard.dispatch.state import RunRecord
from mainboard.dispatch.vocabulary import JobState, Resources
from mainboard.doctor import Doctor
from mainboard.engines.compile import Provisioner
from mainboard.manifest import Manifest
from mainboard.monitor import Monitor
from mainboard.scaffold import Scaffold

from .dispatch.backends.support import BareBackend

if TYPE_CHECKING:
    from pathlib import Path

_GOLD = "gold"
_MIYABI_G = "miyabi-g"
_REMOTE_ROOT = "/work/xg25g007/x10537/projects"

# What a local install asks its provisioner for, the environment it compiled and the module
# stack it activated, or the workspace root it was pointed at.
type Provisioned = tuple[str, Path | tuple[str, dict[str, str]] | tuple[str, bool]]


class Replaced(Exception):
    """What the injected exec seam raises, standing in for this process being replaced."""


class FakeConnection:
    """The one ssh connection a bound board opens, answering every bound command with `reply`."""

    def __init__(self, reply: str) -> None:
        """reply: what running the staged line answers with."""
        self.reply = reply

    def __call__(self) -> str:
        return self.reply

    def __enter__(self) -> FakeConnection:
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        fault: BaseException | None,
        trace: TracebackType | None,
    ) -> bool:
        return False

    def __getitem__(self, name: str) -> FakeConnection:
        return self


class FakeProvisioner:
    """A `Provisioner` double recording what a local install compiled and activated."""

    calls: list[Provisioned] = []

    def __init__(self, root: Path, manifest: Manifest) -> None:
        FakeProvisioner.calls.append(("init", root))

    def activate(self, env: str, *, modules: dict[str, str]) -> str:
        FakeProvisioner.calls.append(("activate", (env, dict(modules))))
        return f"/repo/.mainboard/{env}-activate.sh"

    def provision(self, env: str, *, resolve: bool) -> None:
        FakeProvisioner.calls.append(("provision", (env, resolve)))


class FakeCloud(ProviderBackend, Account, Delivery, LogSource):
    """A registered `fakecloud`-kind backend carrying every capability, so nothing is refused.

    Module-level so it registers exactly once.
    """

    name = "fakecloud"
    submitted: list[str] = []
    cancelled: list[str] = []

    def cancel(self, handle: str) -> None:
        FakeCloud.cancelled.append(handle)

    def deliver(self, handle: str, *, path: str) -> None:
        return None

    def logs(self, handle: str) -> str:
        return "cloud log"

    def standing(self) -> Standing:
        return Standing(keyed=True, note="a fake cloud is always paid up")

    def state(self, handle: str) -> JobState:
        return JobState(handle=handle, state="finished", exit_code=0, verdict="ok")

    def submit(self, plan: ExecutionPlan, command: str, resources: Resources) -> str:
        FakeCloud.submitted.append(command)
        return "cloud-1"


@pytest.fixture
def installed(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Board]:
    """A board on its own workspace where `default` is provisioned and a stub `pixi` is on PATH.

    pixi stamps the fingerprint only once an install completes, so writing it is what makes the
    workspace look provisioned, and the stub is what a shell would have been replaced by. The
    workspace is this test's own rather than the shared station, since both artifacts would
    otherwise outlive it and make a later test read an environment nobody installed.
    """
    fingerprint = workspace / ".mainboard" / ".pixi" / "envs" / "default" / "conda-meta"
    fingerprint.mkdir(parents=True)
    (fingerprint / ".pixi-environment-fingerprint").write_text("installed\n")
    bindir = workspace / "bin"
    bindir.mkdir()
    binary = bindir / "pixi"
    binary.write_text("#!/bin/sh\n")
    binary.chmod(0o755)
    monkeypatch.chdir(workspace)
    with local.env(PATH=f"{bindir}{os.pathsep}{local.env['PATH']}"):
        yield Board(workspace)


def cloud_job(board: Board, monkeypatch: pytest.MonkeyPatch, *, fetch: str = "") -> ProviderJob:
    """A submitted `ProviderJob` routed to the shared `fakecloud`-kind backend."""
    FakeCloud.submitted = []
    FakeCloud.cancelled = []
    manifest = board.manifest.model_copy(
        update={
            "hosts": {
                **board.manifest.hosts,
                "cloudbox": board.manifest.profile(_GOLD).model_copy(update={"kind": "fakecloud"}),
            }
        }
    )
    monkeypatch.setitem(board.shared, "manifest", manifest)
    monkeypatch.setitem(board.shared, "resolver", None)
    submitted = board.on("cloudbox").submit("python train.py", mem_gb=8, fetch=fetch or None)
    assert isinstance(submitted, ProviderJob)
    return submitted


def test_on_binds_the_host_and_shares_the_loaded_manifest(board: Board) -> None:
    assert board.local and board.host == "local"
    bound = board.on(_MIYABI_G)
    assert bound.host == _MIYABI_G and not bound.local
    assert bound.manifest is board.manifest
    assert bound.dispatcher is board.dispatcher
    assert bound.resolver is board.resolver
    assert bound.plan().containerized


def test_one_shared_subsystem_is_built_once_however_many_threads_ask_at_once(
    board: Board,
) -> None:
    """Two threads reaching an unbuilt subsystem together would otherwise each build one.

    A second dispatcher means a second dispatch cache, and a second cache is a second SQLite
    connection owned by whichever thread happened to win, so the build is locked rather than
    merely memoized. The waiting thread is let into `once` only once the other is already
    inside its builder, which is the interleaving that used to produce two.
    """
    building = Event()

    def slow() -> str:
        building.set()
        sleep(0.05)
        return "built"

    thread = Thread(target=lambda: board.once("thing", slow))
    thread.start()
    assert building.wait(timeout=5.0)
    mine = board.once("thing", object)
    thread.join()
    assert mine is board.shared["thing"]
    assert board.on(_GOLD).once("thing", object) is mine


@pytest.mark.parametrize(
    ("accessor", "kind"),
    [
        ("compute", Survey),
        ("deps", Dependencies),
        ("doctor", Doctor),
        ("fleet", Fleet),
        ("monitor", Monitor),
        ("scaffold", Scaffold),
    ],
)
def test_the_board_hands_out_each_subsystem_bound_to_this_workspace(
    board: Board, accessor: str, kind: type
) -> None:
    """Every verb reaches its subsystem through the one addressable interface.

    The host-independent ones answer the same whatever host the board is pivoted onto, since a
    survey, a sweep and a dependency belong to the workspace rather than to one machine.
    """
    subsystem = getattr(board.on(_GOLD), accessor)()
    assert isinstance(subsystem, kind)
    assert subsystem.board.manifest is board.manifest


@pytest.mark.parametrize(("command", "code"), [("true", 0), ("false", 1)])
def test_a_local_run_executes_the_wrapped_line_and_answers_with_its_exit_code(
    board: Board, command: str, code: int
) -> None:
    assert board.run(command) == code


def test_run_hands_a_declared_task_to_pixi_and_anything_else_to_the_shell(
    board: Board, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`test` is a task in the fixture manifest, so the shell never sees that word at all."""
    staged: list[str] = []

    def capture(
        plan: ExecutionPlan,
        root: str,
        *,
        command: str,
        containerize: Callable[[list[str]], list[str]] | None = None,
    ) -> str:
        staged.append(command)
        return "true"

    monkeypatch.setattr("mainboard.board.wrap", capture)

    board.run("test --quiet")
    board.run("pytest --quiet")

    assert staged == [
        "pixi run --manifest-path .mainboard/pixi.toml --frozen -e default test --quiet",
        "pytest --quiet",
    ]


def test_attest_publishes_one_reading_of_this_machine_into_the_streams_receipts(
    board: Board,
) -> None:
    """The synchronous once-only twin of the sampler, read on the node that does the work."""
    board.attest("smoke-1", job="gold-1")
    (line,) = board.receipts("smoke-1").replay()
    assert line.topic is Topic.ATTESTED
    assert (line.batch, line.job) == ("smoke-1", "gold-1")
    assert "idle" in line.data and "gpu_pct" in line.data


def test_remote_root_comes_from_the_profile_or_refuses(board: Board) -> None:
    assert board.on(_MIYABI_G).remote_root() == _REMOTE_ROOT
    with pytest.raises(MissionError, match=r"set \[hosts.gold\] root"):
        board.on(_GOLD).remote_root()


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        (
            {"attempt": 2},
            {"queue": "debug-g", "walltime": "00:30:00", "mem": 100, "account": "xg25g007"},
        ),
        (
            {"queue": "short-g", "mem_gb": 64, "attempt": 1},
            {"queue": "short-g", "walltime": "00:30:00", "mem": 64, "account": "xg25g007"},
        ),
    ],
    ids=["profile defaults with the expression evaluated", "overrides beat the defaults"],
)
def test_submit_resolves_the_hosts_declared_resources(
    board: Board, monkeypatch: pytest.MonkeyPatch, given: dict[str, int | str], expected: dict
) -> None:
    seen: dict[str, str | int | bool] = {}

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
    job = board.on(_MIYABI_G).submit("python -m exp.run", **given)
    assert isinstance(job, Job) and job.handle.id == "77"
    assert seen == {**expected, "containerized": True, "root": _REMOTE_ROOT}


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ({"env": "serving", "resolve": True}, ("serving", {})),
        ({"profile": _MIYABI_G}, ("default", {"singularity": "4.2.1"})),
    ],
    ids=["a named environment", "the module stack of a named profile"],
)
def test_installing_here_provisions_and_activates_in_place(
    board: Board, monkeypatch: pytest.MonkeyPatch, given: dict[str, str | bool], expected: tuple
) -> None:
    FakeProvisioner.calls = []
    monkeypatch.setattr("mainboard.board.Provisioner", FakeProvisioner)
    setup = board.install(**given)
    assert ("provision", (expected[0], given.get("resolve", False))) in FakeProvisioner.calls
    assert ("activate", expected) in FakeProvisioner.calls
    assert setup.host == "local"
    assert setup.installer == "in-place"
    assert setup.activate.endswith(f"{expected[0]}-activate.sh")
    assert setup.tool


@pytest.mark.parametrize(
    ("host", "env", "expected"),
    [
        (_MIYABI_G, "serving", ("serving", _REMOTE_ROOT)),
        (_GOLD, "", ("serving", "")),
    ],
    ids=["a named environment on a rooted host", "the environment the profile itself names"],
)
def test_installing_a_host_onboards_it_with_the_lock_this_workspace_solved(
    board: Board, monkeypatch: pytest.MonkeyPatch, host: str, env: str, expected: tuple[str, str]
) -> None:
    """gold declares `env = "serving"`, so setting gold up must not fall back to default."""
    seen: dict[str, str | int | bool] = {}
    report = HostSetup(host=host, root="/repo", installer="uv")

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

        def run(self, *, sync_only: bool = False) -> HostSetup:
            seen["sync_only"] = sync_only
            return report

    monkeypatch.setattr("mainboard.board.Onboarding", FakeOnboarding)
    assert board.on(host).install(env) is report
    assert seen == {
        "host": host,
        "root": expected[1],
        "env": expected[0],
        "artifact": (".mainboard/pixi.toml", ".mainboard/pixi.lock", ".mainboard/state.toml"),
        "resolve": False,
        "containerized": False,
        "sync_only": False,
    }


def test_sync_only_reaches_the_onboarding_and_is_refused_on_this_machine(
    board: Board, monkeypatch: pytest.MonkeyPatch
) -> None:
    """This machine has no onboarding to shortcut, so the flag refuses rather than no-op."""
    seen: dict[str, bool] = {}
    report = HostSetup(host=_GOLD, root="/repo", installer="uv")

    class FakeOnboarding:
        def __init__(self, dispatcher, plan, *, root, artifact, resolve, watch):
            pass

        def run(self, *, sync_only: bool = False) -> HostSetup:
            seen["sync_only"] = sync_only
            return report

    monkeypatch.setattr("mainboard.board.Onboarding", FakeOnboarding)
    assert board.on(_GOLD).install(sync_only=True) is report
    assert seen == {"sync_only": True}

    with pytest.raises(MissionError, match="--sync-only"):
        board.install(sync_only=True)


def test_shell_replaces_this_process_with_a_frozen_pixi_shell(installed: Board) -> None:
    """The terminal goes to `pixi shell`, pinned to the workspace and forbidden to solve."""
    seen: list[tuple[str, list[str]]] = []

    def replace(path: str, argv: list[str], env: Mapping[str, str]) -> NoReturn:
        seen.append((path, argv))
        raise Replaced

    with pytest.raises(Replaced):
        installed.shell(replace=replace)

    binary = str(installed.root / "bin" / "pixi")
    manifest = str(installed.root / ".mainboard" / "pixi.toml")
    argv = [binary, "shell", "--manifest-path", manifest, "--frozen", "-e", "default"]
    assert seen == [(binary, argv)]


def test_shell_carries_the_workspace_floors_into_the_replacing_process(
    installed: Board, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Replacing this process drops the bound environment, so the floors travel explicitly.

    A shell is the one path that execs rather than spawns, so a floor that only rides on the
    plumbum command reaches every child except this one, which is how a host that cannot
    present the virtual package fails on `shell` alone.
    """
    monkeypatch.delenv("CONDA_OVERRIDE_CUDA", raising=False)
    pixi = Provisioner(installed.root, installed.manifest).pixi
    pixi.manifest.write_text('[workspace]\nplatforms = [{ cuda = "13.0" }]\n', encoding="utf-8")
    seen: list[Mapping[str, str]] = []

    def replace(path: str, argv: list[str], env: Mapping[str, str]) -> NoReturn:
        seen.append(env)
        raise Replaced

    with pytest.raises(Replaced):
        installed.shell(replace=replace)

    assert seen[0]["CONDA_OVERRIDE_CUDA"] == "13.0"
    assert seen[0]["PATH"] == os.environ["PATH"]


@pytest.mark.parametrize(
    ("host", "env", "refusal"),
    [
        ("local", "serving", "Run `mainboard install serving`"),
        (_GOLD, "", "this machine only"),
    ],
    ids=["an environment nothing provisioned", "a board bound to another machine"],
)
def test_shell_refuses_what_it_cannot_hand_the_terminal_to(
    board: Board, host: str, env: str, refusal: str
) -> None:
    """A shell on the system interpreter is the failure the refusal exists to prevent."""
    with pytest.raises(MissionError, match=refusal):
        board.on(host).shell(env)


def interacting() -> tuple[list[list[str]], Callable[[str, list[str]], NoReturn]]:
    """A recording exec seam, plus the argv list it appends each replaced process to."""
    seen: list[list[str]] = []

    def replace(path: str, argv: list[str]) -> NoReturn:
        assert path == "ssh"
        seen.append(argv)
        raise Replaced

    return seen, replace


def test_interact_hands_an_ssh_host_terminal_to_that_hosts_own_tool(
    board: Board, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An ssh box is already the machine the work runs on, so nothing is allocated for it."""
    rooted = board.manifest.model_copy(
        update={
            "hosts": {
                **board.manifest.hosts,
                _GOLD: board.manifest.profile(_GOLD).model_copy(update={"root": "/home/p/lab"}),
            }
        }
    )
    monkeypatch.setitem(board.shared, "manifest", rooted)
    monkeypatch.setitem(board.shared, "resolver", None)
    seen, replace = interacting()

    with pytest.raises(Replaced):
        board.on(_GOLD).interact(replace=replace)
    [argv] = seen
    assert argv[:3] == ["ssh", "-t", _GOLD]
    assert argv[3].startswith("bash -lc ")
    assert "cd /home/p/lab" in argv[3]
    assert argv[3].endswith("mainboard shell serving'")

    seen.clear()
    with pytest.raises(Replaced):
        board.on(_GOLD).interact("pwd", replace=replace)
    assert seen[0][3].endswith("mainboard run --env serving -- pwd'")


def test_interact_asks_a_queued_host_for_an_allocation_before_the_terminal(
    board: Board,
) -> None:
    """A PBS terminal belongs on an allocated node, never on the login node that asked."""
    seen, replace = interacting()
    with pytest.raises(Replaced):
        board.on(_MIYABI_G).interact(replace=replace)
    staged = seen[0][3]
    assert f"cd {_REMOTE_ROOT}" in staged
    assert "module load singularity/4.2.1" in staged
    assert staged.endswith("qsub -I -q debug-g -l walltime=00:30:00 -W group_list=xg25g007'")


def test_interact_prefers_the_declared_interactive_queue_over_the_batch_one(
    board: Board, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A site that routes interactive work elsewhere says so once, on the profile."""
    profile = board.manifest.profile(_MIYABI_G)
    routed = board.manifest.model_copy(
        update={
            "hosts": {
                **board.manifest.hosts,
                _MIYABI_G: profile.model_copy(
                    update={
                        "defaults": profile.defaults.model_copy(
                            update={"interact_queue": "interact-g"}
                        )
                    }
                ),
            }
        }
    )
    monkeypatch.setitem(board.shared, "manifest", routed)
    monkeypatch.setitem(board.shared, "resolver", None)
    seen, replace = interacting()

    with pytest.raises(Replaced):
        board.on(_MIYABI_G).interact(replace=replace)
    assert "qsub -I -q interact-g" in seen[0][3]

    seen.clear()
    with pytest.raises(Replaced):
        board.on(_MIYABI_G).interact(queue="short-g", walltime="06:00:00", replace=replace)
    assert "qsub -I -q short-g -l walltime=06:00:00" in seen[0][3]


@pytest.mark.parametrize(
    ("host", "options", "refusal"),
    [
        ("local", {}, "needs a host"),
        ("cloudbox", {}, "hands out no terminal"),
        (_MIYABI_G, {"walltime": "09:00:00"}, "exceeds the 'debug-g' ceiling"),
    ],
    ids=["this machine", "a rented instance", "a walltime the queue would reject"],
)
def test_interact_refuses_what_it_cannot_hand_a_terminal_to(
    board: Board, monkeypatch: pytest.MonkeyPatch, host: str, options: dict[str, str], refusal: str
) -> None:
    """The scheduler's own rejection arrives minutes later; this one arrives before the ssh."""
    rented = board.manifest.model_copy(
        update={
            "hosts": {
                **board.manifest.hosts,
                "cloudbox": board.manifest.profile(_GOLD).model_copy(
                    update={"kind": "fakecloud", "root": "/rented"}
                ),
            }
        }
    )
    monkeypatch.setitem(board.shared, "manifest", rented)
    monkeypatch.setitem(board.shared, "resolver", None)
    with pytest.raises(MissionError, match=refusal):
        board.on(host).interact(**options)


def test_a_scheduler_job_delegates_every_verb_to_its_host_and_dispatcher(
    board: Board, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, str]] = []
    handle = Handle(id="9", host=_MIYABI_G, root="/work/p", kind="pbs")
    verdict = Verdict(verdict="ok")
    probed = JobState(handle="9", state="R", verdict="running")

    class FakeScheduler:
        def cancel(self, remote: FakeConnection, root: str, *, handle: str) -> None:
            calls.append(("cancel", handle))

        def logs(self, remote: FakeConnection, root: str, *, handle: str) -> str:
            calls.append(("logs", handle))
            return "the log"

    # The recorded kind picks the backend, never the host's currently declared profile, so a
    # host whose kind changed under a live job still has that job killed the way it was taken.
    monkeypatch.setattr(
        registry.SCHEDULERS, "select", lambda kind, default: FakeScheduler(), raising=False
    )
    monkeypatch.setattr("mainboard.board.connection", lambda host: FakeConnection(""))
    monkeypatch.setattr(
        board.dispatcher, "await_many", lambda handles, **kw: {handles[0]: verdict}
    )
    monkeypatch.setattr(board.dispatcher, "probe", lambda asked: probed)
    monkeypatch.setattr(board.dispatcher, "state", lambda asked: probed)
    monkeypatch.setattr(board.dispatcher, "fetch", lambda asked, **kw: calls.append(("pull", "9")))

    job = Job(board.on(_MIYABI_G), handle)
    assert job.logs() == "the log"
    job.kill()
    job.pull()
    assert job.wait() is verdict
    assert job.state() is probed
    assert job.poll() is probed  # the same read with the blip left unabsorbed
    assert calls == [("logs", "9"), ("cancel", "9"), ("pull", "9")]


def test_a_bound_board_reads_facts_and_runs_commands_over_one_connection(
    board: Board, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A remote fact read parses the last JSON line out of whatever the login shell said first."""
    payload = HostFacts(schema_version=1, hostname="fake-remote").model_dump_json()
    monkeypatch.setattr(
        "mainboard.board.connection", lambda host: FakeConnection(f"module chatter\n{payload}\n")
    )
    monkeypatch.setattr("mainboard.board._streamed", lambda command: 7)
    bound = board.on(_MIYABI_G)
    assert bound.facts().hostname == "fake-remote"
    assert bound.run("true", container="none") == 7


def test_a_provider_submit_is_recorded_so_a_later_process_rebuilds_the_rental(
    board: Board, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without the record no later process could settle the run, and the rental would bill on."""
    submitted = cloud_job(board, monkeypatch, fetch="results/run")
    assert submitted.backend.submitted == ["python train.py"]
    record = board.dispatcher.cache.run(submitted.handle.id)
    assert (record.target, record.kind) == ("cloudbox", "fakecloud")
    assert (record.script, record.fetch_path) == ("python train.py", "results/run")
    assert record.verdict is None
    rebuilt = board.job(submitted.handle.id)
    assert isinstance(rebuilt, ProviderJob)
    assert rebuilt.handle == submitted.handle
    assert rebuilt.backend.name == "fakecloud"


def test_a_provider_job_delegates_to_its_backend_and_ends_the_rental_on_a_wait(
    board: Board, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A caller that blocked on the run is the last thing between it and an idle meter."""
    job = cloud_job(board, monkeypatch, fetch="results/")
    states = iter(["running", "running", "ok"])
    job.backend.state = lambda handle: JobState(handle=handle, state="x", verdict=next(states))
    assert job.poll().verdict == "running"
    verdict = job.wait(poll=lambda seconds: None)
    assert verdict.ok and job.logs() == "cloud log"
    assert job.backend.cancelled == [job.handle.id]
    job.kill()
    job.pull()


@pytest.mark.parametrize(
    ("fetch", "fault", "refusal"),
    [
        ("results/", MissionError, r"bare backend keeps no logs; read bare-1\.log"),
        ("results/", MissionError, "does not implement Delivery"),
        ("", LookupError, "no fetch path"),
    ],
    ids=["logs it never keeps", "a delivery it never had", "a path nobody recorded"],
)
def test_a_provider_job_refuses_a_capability_its_backend_never_had(
    fetch: str, fault: type[Exception], refusal: str
) -> None:
    """The absence is discovered before the call, and answered with the backend's own advice."""
    handle = Handle(id="bare-1", host="cloudbox", root="", kind="bare", fetch_path=fetch or None)
    job = ProviderJob(BareBackend(), handle)
    job.kill()
    assert job.wait(poll=lambda seconds: None).ok
    asked = job.logs if "logs" in refusal else job.pull
    with pytest.raises(fault, match=refusal):
        asked()


@pytest.mark.parametrize(
    ("backend", "expected"),
    [
        pytest.param(BareBackend(), "", id="a-provider-that-keeps-no-log-at-all"),
        pytest.param(FakeCloud(), "cloud log", id="a-provider-that-hands-its-log-over"),
    ],
)
def test_a_transcript_is_the_tolerant_twin_of_logs_so_a_settle_never_dies_over_one(
    backend: ProviderBackend, expected: str
) -> None:
    """The settle that captures output runs before the release, and must not fail the sweep."""
    handle = Handle(id="c-1", host="cloudbox", root="", kind=backend.name)
    assert ProviderJob(backend, handle).transcript() == expected


def test_a_provider_that_will_not_hand_its_log_over_costs_a_transcript_and_nothing_else(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cloud = FakeCloud()
    monkeypatch.setattr(
        cloud, "logs", lambda handle: (_ for _ in ()).throw(MissionError("vast refused logs"))
    )
    handle = Handle(id="c-2", host="cloudbox", root="", kind="fakecloud")
    assert ProviderJob(cloud, handle).transcript() == ""


def test_job_rebuilds_a_dispatched_run_from_the_cache(
    board: Board, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fresh process addresses an already-running job the way the one that submitted it did."""
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
    assert job.handle.root == _REMOTE_ROOT
    assert job.handle.fetch_path == "results/run"
    assert job.board.host == _MIYABI_G
