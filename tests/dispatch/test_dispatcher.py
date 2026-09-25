import inspect
import os
import stat
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing, nullcontext
from pathlib import Path
from shutil import which
from threading import Barrier
from typing import TYPE_CHECKING

import pytest
from plumbum import local
from plumbum.commands.processes import ProcessExecutionError

from mainboard import Board, ExecutionPlan, MissionError
from mainboard.dispatch import (
    Dispatcher,
    GitignoreFilter,
    Handle,
    HostSetup,
    Shipment,
    SyncLock,
    Verdict,
)
from mainboard.dispatch import dispatcher as dispatch_module
from mainboard.dispatch.jobs import JobSpec
from mainboard.dispatch.provenance import SourceTree
from mainboard.dispatch.provenance import listing as source_listing
from mainboard.dispatch.schedulers import HostUnreachable, registry
from mainboard.dispatch.snapshots import CLOSURE, Snapshots
from mainboard.dispatch.state import Cache
from mainboard.dispatch.sync import binary
from mainboard.dispatch.vocabulary import POLL_SECONDS, JobState, Request, Resources
from mainboard.manifest import Container, Defaults, HostProfile, QueuePolicy
from mainboard.runtime.job import PrefixActivation, ToolCall, WorkspaceActivation

from ..support import Lab
from .support import (
    RecordingScheduler,
    cache,
    machine_with,
    pins_on_this_host,
    plan,
    recorded,
    run_record,
    setgid_inherits,
)

if TYPE_CHECKING:
    from mainboard.dispatch.transport import Machine, SshTransport

_CONTAINERIZED = {
    "profile": HostProfile(kind="ssh", root="/repo", container="ngc", sync={"include": ["src"]}),
    "container": Container(image="nvcr.io/nvidia/pytorch:25.06-py3"),
}


class _StubStrategy:
    """A `Strategy`-shaped double that always resolves to one canned scheduler."""

    def __init__(self, scheduler: RecordingScheduler) -> None:
        self.scheduler = scheduler

    def select(self, kind: str, default: str | None = None) -> RecordingScheduler:
        del kind, default
        return self.scheduler


def shipped(dispatcher: Dispatcher, command: str, imports: tuple[str, ...] = ()) -> Shipment:
    """`command` as a board ships it to the dispatcher: the mirror, under the tree's provenance."""
    return Shipment.of_command(command, source=dispatcher.source(command), imports=imports)


@pytest.mark.parametrize("entry", ["submit", "allocating"])
@pytest.mark.parametrize("missing", [False, True])
def test_direct_dispatch_entries_refuse_stale_registration_before_external_work(
    lab: Lab, monkeypatch: pytest.MonkeyPatch, entry: str, missing: bool
) -> None:
    board = Board(lab.root)
    shipment = board.shipment(f"{Lab.JOB}::app", board.plan())
    node = lab.root / "research/camp/experiments/node/node.md"
    if missing:
        node.unlink()
    else:
        node.write_text("changed after seal\n")
    dispatcher = board.dispatcher
    monkeypatch.setattr(dispatcher, "rsync_up", lambda *a, **kw: pytest.fail("transport reached"))
    monkeypatch.setattr(
        dispatcher.cache, "reserve", lambda *a, **kw: pytest.fail("creation reserved")
    )
    with pytest.raises((MissionError, FileNotFoundError)):
        if entry == "submit":
            dispatcher.submit(
                plan(), "/repo", script=Lab.JOB, args=(), resources=Resources(), shipment=shipment
            )
        else:
            dispatcher.allocating(plan(), shipment, Resources(), evidence="not_started")


@pytest.fixture
def backend(monkeypatch: pytest.MonkeyPatch) -> RecordingScheduler:
    """Pin the backend, the connection, git and the clock, every seam a dispatch reaches."""
    scheduler = RecordingScheduler()
    monkeypatch.setattr(dispatch_module, "pick", lambda profile: scheduler)
    monkeypatch.setattr(registry, "SCHEDULERS", _StubStrategy(scheduler))
    monkeypatch.setattr(dispatch_module, "connection", lambda host: machine_with())
    monkeypatch.setattr(dispatch_module, "sleep", lambda seconds: None)
    return scheduler


@pytest.fixture
def dispatcher(workdir: Path, backend: RecordingScheduler) -> Dispatcher:
    """A dispatcher whose mirror only records what it was asked to ship, on `instance.shipped`.

    The double answers the allowlist the real mirror would have shipped, since that file set is
    what the snapshot of the mirror copies.
    """
    del backend
    instance = Dispatcher(cache=cache(), sync=GitignoreFilter(workdir))
    instance.shipped: list[tuple[str, ...]] = []
    instance.required: list[list[list[str]]] = []

    def mirror(execution: ExecutionPlan, root: str, **kwargs: object) -> list[str]:
        del execution, root
        extra = kwargs.get("extra", ())
        instance.shipped.append(tuple(extra) if isinstance(extra, tuple | list) else ())
        needed = kwargs.get("required", ())
        instance.required.append(
            [list(group) for group in needed] if isinstance(needed, tuple | list) else []
        )
        return ["src"]

    instance.rsync_up = mirror
    return instance


@pytest.mark.parametrize(
    ("verdict", "exit_code", "code", "ok"),
    [
        ("ok", 0, 0, True),
        ("failed", 1, 1, False),
        ("running", None, 2, False),
        ("vanished", None, 3, False),
        ("unknown", None, 3, False),
    ],
)
def test_a_verdict_projects_onto_the_process_exit_code_a_caller_branches_on(
    verdict: str, exit_code: int | None, code: int, ok: bool
) -> None:
    projected = Verdict(verdict=verdict, exit_code=exit_code)
    assert (projected.code, projected.ok) == (code, ok)


def test_a_numeric_scheduler_id_is_stored_as_text_wherever_a_handle_travels() -> None:
    """pueue hands out small integers, and a caller reading one back as a number fails deep."""
    handle = Handle(id=42, host="gold", root="/repo", kind="ssh", fetch_path="out/")
    assert (handle.id, handle.fetch_path) == ("42", "out/")
    assert JobState(handle=42, verdict="running").handle == "42"


def test_run_renders_a_job_script_ships_it_and_hands_back_a_pollable_handle(
    dispatcher: Dispatcher, backend: RecordingScheduler, workdir: Path
) -> None:
    resources = Resources(gpus=4, walltime="01:00:00", queue="gen-S", mem_gb=240)
    handle = dispatcher.run(
        plan(),
        shipped(dispatcher, "python -m foo --shard 3"),
        root="/repo",
        resources=resources,
        fetch="out/",
    )
    assert (handle.id, handle.host, handle.kind, handle.fetch_path) == (
        "H1",
        "gold",
        "ssh",
        "out/",
    )
    [(_root, script, args)] = [call for name, call in backend.calls if name == "submit"]
    assert script.startswith(".mainboard-jobs/")
    assert args == ()
    [(staged,)] = dispatcher.shipped
    assert Snapshots.script(staged) == script
    assert (workdir / staged).is_file()
    assert backend.submit_resources == resources


def test_run_renders_the_job_script_against_the_plans_own_environment(
    dispatcher: Dispatcher, workdir: Path
) -> None:
    """A job queued for `serving` must activate serving, not whatever was installed last."""
    dispatcher.run(
        plan(env="serving"),
        shipped(dispatcher, "python -m foo"),
        root="/repo",
        resources=Resources(),
    )
    [generated] = (workdir / ".mainboard" / "dispatch" / "jobs").glob("job-*.sh")
    job = recorded(generated.read_text())
    # The job activates through the snapshot this dispatch pinned, whose `.mainboard` is a
    # symlink back to the mirror, so it gets the mirror's environment out of a tree whose code
    # no later sync can rewrite.
    pinned = dispatcher.pinned("/repo", source=dispatcher.source())
    assert pinned.startswith("/repo/.mainboard/dispatch/sources/")
    assert job.activation == WorkspaceActivation(
        script=f"{pinned}/.mainboard/activate-serving.sh",
        prefix=f"{pinned}/.mainboard/envs/serving/.pixi/envs/serving",
        refusal=job.activation.refusal,
    )
    assert "mainboard setup gold --env serving" in job.activation.refusal


def test_run_on_a_pbs_host_with_no_resolved_walltime_fails_before_any_sync(
    dispatcher: Dispatcher, backend: RecordingScheduler
) -> None:
    """No declared default and no resolved walltime is a clear error, never a site constant."""
    pbs = plan(profile=HostProfile(kind="pbs", root="/repo", sync={"include": ["src"]}))
    with pytest.raises(ValueError, match="explicit walltime"):
        dispatcher.run(
            pbs, shipped(dispatcher, "python -m foo"), root="/repo", resources=Resources()
        )
    assert dispatcher.shipped == []
    assert backend.calls == []


def test_run_containerized_wraps_the_command_via_the_builder_or_refuses_without_one(
    dispatcher: Dispatcher, backend: RecordingScheduler, workdir: Path
) -> None:
    containerized = plan(**_CONTAINERIZED)
    with pytest.raises(LookupError, match="no container argv builder"):
        dispatcher.run(
            containerized,
            shipped(dispatcher, "python -m foo"),
            root="/repo",
            resources=Resources(),
        )
    dispatcher.run(
        containerized,
        shipped(dispatcher, "python -m foo"),
        root="/repo",
        resources=Resources(),
        containerize=lambda inner: ["apptainer", "exec", "image.sif", *inner],
    )
    [(_root, script, _args)] = [call for name, call in backend.calls if name == "submit"]
    job = recorded((workdir / ".mainboard/dispatch/jobs" / Path(script).name).read_text())
    assert job.container == ("apptainer", "exec", "image.sif", "bash", "-c", "python -m foo")


def test_submit_admits_the_request_before_a_single_ssh_connection(
    dispatcher: Dispatcher, backend: RecordingScheduler
) -> None:
    host = HostProfile(
        kind="ssh",
        root="/repo",
        sync={"include": ["src"]},
        queues={"short-g": QueuePolicy(max_walltime="00:10:00")},
    )
    with pytest.raises(MissionError, match="exceeds the 'short-g' ceiling"):
        dispatcher.submit(
            plan(profile=host),
            "/repo",
            script="job.sh",
            args=(),
            resources=Resources(queue="short-g", walltime="08:00:00"),
        )
    assert dispatcher.shipped == []
    assert backend.calls == []


def test_submit_records_content_identity_without_git(
    dispatcher: Dispatcher,
    backend: RecordingScheduler,
) -> None:
    handle = dispatcher.submit(
        plan(), "/repo", script="train.sh", args=("--x", "1"), resources=Resources()
    )
    [run] = dispatcher.cache.recent(10)
    assert (run.handle, run.target, run.git_sha, run.dirty) == (handle, "gold", "", None)
    assert run.args == "--x 1" and run.script == "train.sh"
    assert not run.commit and len(run.digest) == 64
    assert run.source == f"sha256-{run.digest}"


def test_direct_script_submission_keeps_the_prepared_path_and_arguments(
    dispatcher: Dispatcher, backend: RecordingScheduler, workdir: Path
) -> None:
    script = workdir / "script with spaces.sh"
    script.write_text("#!/bin/bash\nexit 0\n")
    args = ("--label", "a b")
    handle = dispatcher.submit(
        plan(), "/repo", script=str(script), args=args, resources=Resources()
    )
    record = dispatcher.cache.run(handle)
    [(_, prepared, submitted_args)] = [call for name, call in backend.calls if name == "submit"]
    assert Snapshots.script(record.script) == prepared
    assert record.script.startswith(".mainboard/dispatch/jobs/")
    assert (workdir / record.script).read_bytes() == script.read_bytes()
    assert submitted_args == args and record.args == "--label 'a b'"


@pins_on_this_host
def test_submission_reaches_the_scheduler_only_after_the_wrapper_is_frozen(
    dispatcher: Dispatcher,
    backend: RecordingScheduler,
    workdir: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Use real staging, rsync, and pinning; the scheduler never executes the script."""
    mirror = tmp_path / "remote"
    mirror.mkdir()
    (workdir / "src").mkdir(exist_ok=True)
    authored = workdir / "handwritten script.sh"
    payload = b"#!/bin/sh\r\n# exact byte: \xff\r\n"
    authored.write_bytes(payload)
    authored.chmod(0o750)

    def transfer(*args, **kwargs):
        local["rsync"]["-a", "--exclude=remote", str(workdir) + "/", str(mirror) + "/"]()
        return ["src"]

    def submit(remote, root, *, script, args, resources):
        assert script.startswith(".mainboard-jobs/")
        frozen = Path(root) / script
        assert frozen.read_bytes() == payload and not frozen.is_symlink()
        assert args == ("--label", "a b")
        return "frozen-control"

    monkeypatch.setattr(dispatcher, "rsync_up", transfer)
    monkeypatch.setattr(dispatcher, "_verify", lambda *a, **kw: None)
    monkeypatch.setattr(dispatcher, "_prime", lambda *a, **kw: None)
    monkeypatch.setattr(dispatch_module, "connection", lambda host: nullcontext(local))
    monkeypatch.setattr(backend, "submit", submit)
    assert (
        dispatcher.submit(
            plan(),
            str(mirror),
            script=str(authored),
            args=("--label", "a b"),
            resources=Resources(),
        )
        == "frozen-control"
    )
    assert authored.read_bytes() == payload and stat.S_IMODE(authored.stat().st_mode) == 0o750


def test_a_dispatch_runs_from_a_snapshot_of_the_mirror_and_never_from_the_mirror_itself(
    dispatcher: Dispatcher, backend: RecordingScheduler, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The fault this exists for: a later sync of another tree rewrote the code under live jobs."""
    machine = machine_with()
    monkeypatch.setattr(dispatch_module, "connection", lambda host: machine)
    handle = dispatcher.run(
        plan(),
        shipped(dispatcher, "python -m foo"),
        root="/repo",
        resources=Resources(),
        fetch="out/raw",
    )
    pinned = dispatcher.pinned("/repo", source=dispatcher.source())
    [(root, _script, _args)] = [call for name, call in backend.calls if name == "submit"]
    assert root == pinned
    # The handle still names the mirror: the logs, the exit artifact and the results live there
    # and stay there once the snapshot is pruned.
    assert handle.root == "/repo"
    # The tree is materialised from the file set that transfer shipped, before anything is
    # queued into it, and the declared results path is linked back to the mirror.
    [built] = [line for line in machine.lines if "--link-dest" in line]
    assert '--link-dest=/repo/ src "$mb_snap"/' in built
    assert f"mb_final={pinned}" in built
    assert 'ln -sfn "$mb_root"/out/raw "$mb_snap"/out/raw' in built
    [run] = dispatcher.cache.recent(10)
    assert run.source == pinned.rsplit("/", maxsplit=1)[-1]


def test_every_job_activates_the_addressed_environment_its_own_tree_names(
    dispatcher: Dispatcher,
    backend: RecordingScheduler,
    workdir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A queued job used to get whatever the one shared prefix was when it started.

    Its tree names a built environment now, by the content of the manifest and lock it was
    dispatched with, so the script builds exactly that one if the host lacks it and then
    activates it and nothing else. The library path is ordered ahead of whatever the machine
    exported, which is how thirty two jobs died importing sqlite3 against `/lib64`'s libstdc++.
    """
    machine = machine_with()
    monkeypatch.setattr(dispatch_module, "connection", lambda host: machine)
    prefix = "/repo/.mainboard/prefixes/default/abcd1234"

    dispatcher.run(
        plan(),
        shipped(dispatcher, "python -m foo"),
        root="/repo",
        resources=Resources(),
        prefix=prefix,
    )

    [(root, script, _args)] = [call for name, call in backend.calls if name == "submit"]
    job = recorded(
        (workdir / ".mainboard/dispatch/jobs" / Path(script).name).read_text(encoding="utf-8")
    )
    # It names the address the dispatch pinned, so a host that reads the shipped artifact as
    # another environment says which two addresses and which two pixis instead of building one
    # beside the one every job of the wave is waiting for. It runs from the mirror, where built
    # environments live, and nothing in the job asks pixi to reconcile anything, which is what
    # made a shared prefix a race in the first place.
    assert job.provide == ToolCall(
        args=(
            "provide",
            "default",
            "--source",
            f"{root}/.mainboard/envs/default",
            "--expect",
            "abcd1234",
        ),
        cwd="/repo",
    )
    assert job.activation == PrefixActivation(
        prefix=prefix, env="default", refusal=job.activation.refusal
    )
    assert f"no completed environment with the expected identity at {prefix}" in (
        job.activation.refusal
    )
    # And the tree points at that environment rather than at the mirror's mutable one.
    [built] = [line for line in machine.lines if "mb_snap=" in line]
    assert f"ln -s {prefix}/.pixi " in built


def test_a_job_imports_the_tree_it_was_pinned_to_and_never_the_mirror(
    dispatcher: Dispatcher,
    backend: RecordingScheduler,
    workdir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A prefix is shared, so the editable install inside it points at the machine's mirror.

    A sync landing between two waves then moves a job's own source while it is queued or
    running, which is the one thing about a dispatched job a shared prefix cannot freeze. The
    job freezes it instead: the pinned tree's import roots go on `PYTHONPATH` ahead of anything
    the environment adds, and nothing the submitting shell exported survives.
    """
    machine = machine_with()
    monkeypatch.setattr(dispatch_module, "connection", lambda host: machine)

    dispatcher.run(
        plan(),
        shipped(
            dispatcher,
            "python -m foo",
            imports=("src", "packages/lab-core/src", ".mainboard/vendor/sample-lib/src"),
        ),
        root="/repo",
        resources=Resources(),
    )

    [(pinned, script, _args)] = [call for name, call in backend.calls if name == "submit"]
    job = recorded(
        (workdir / ".mainboard/dispatch/jobs" / Path(script).name).read_text(encoding="utf-8")
    )
    assert pinned != "/repo"
    # The vendored root rides with the workspace's own: a house package that lives outside the
    # root is compiled inside it, so the tree a job is pinned to carries it like any other.
    assert job.pythonpath == (
        f"{pinned}/src:{pinned}/packages/lab-core/src:{pinned}/.mainboard/vendor/sample-lib/src"
    )


def test_a_sealed_job_ships_its_listing_pins_exactly_that_and_exports_where_it_is(
    dispatcher: Dispatcher,
    backend: RecordingScheduler,
    workdir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The listing rides beside the script, the pin copies what it names, the job reads it."""
    machine = machine_with()
    monkeypatch.setattr(dispatch_module, "connection", lambda host: machine)
    (workdir / "a").mkdir()
    (workdir / "a/run.py").write_text("pass")
    (workdir / "mainboard.toml").write_text("")
    captured, rows = SourceTree(workdir).seal(["a/run.py", "mainboard.toml"])
    sealed = Shipment(
        command="python -m mainboard.jobs.call a/run.py::app -- --x 1",
        spelling="a/run.py::app --x 1",
        source=captured,
        imports=("research/camp", "packages/core/src"),
        listing=source_listing(rows),
        needs=("data/corpus",),
        fetch="a/evidence",
        first_party=("core", "experiments"),
        deferred=("cutoken",),
    )

    handle = dispatcher.run(
        plan(), sealed, root="/repo", resources=Resources(), fetch="a/evidence"
    )

    [(pinned, script, _args)] = [call for name, call in backend.calls if name == "submit"]
    listing = f".mainboard/dispatch/jobs/{sealed.listing_name}"
    assert (workdir / listing).read_text(encoding="utf-8") == sealed.listing
    [(staged, *carried)] = dispatcher.shipped
    assert Snapshots.script(staged) == script
    assert carried == [listing, "a/run.py", "mainboard.toml"]
    # Data resident only on the host must not become a required local transfer. The
    # snapshot still refuses an absent mirror need before the scheduler sees a job.
    assert ["data/corpus"] not in dispatcher.required[0]
    job = recorded((workdir / staged).read_text(encoding="utf-8"))
    assert job.pythonpath == f"{pinned}/research/camp:{pinned}/packages/core/src"
    assert job.variables == {
        "MAINBOARD_SOURCE": captured.identity,
        "MAINBOARD_SOURCE_DIGEST": captured.digest,
        "MAINBOARD_CLOSURE": f"{pinned}/{CLOSURE}",
        "MAINBOARD_FIRST_PARTY": "core:experiments",
        "MAINBOARD_DEFERRED": "cutoken",
    }
    assert job.root == pinned
    [built] = [line for line in machine.lines if "mb_snap=" in line]
    assert f'cut -f1 "$mb_snap/{CLOSURE}" | rsync -aL --filter' in built
    assert "--files-from=-" in built
    assert "hide .card.lock" in built and "protect .card.lock" in built
    assert 'ln -sfn "$mb_root"/data/corpus "$mb_snap"/data/corpus' in built
    assert "for d in" not in built
    [run] = dispatcher.cache.recent(10)
    assert (run.handle, run.dirty, run.source, run.commit) == (
        handle.id,
        None,
        sealed.source.key,
        "",
    )


def test_a_containerized_job_has_no_environment_of_its_own_to_build(
    dispatcher: Dispatcher, backend: RecordingScheduler, workdir: Path
) -> None:
    """What it activates is the image, which no lock on this host describes."""
    dispatcher.run(
        plan(**_CONTAINERIZED),
        shipped(dispatcher, "python -m foo"),
        root="/repo",
        resources=Resources(),
        containerize=lambda argv: ["apptainer", "exec", "img.sif", *argv],
    )
    [(_root, script, _args)] = [call for name, call in backend.calls if name == "submit"]
    job = recorded(
        (workdir / ".mainboard/dispatch/jobs" / Path(script).name).read_text(encoding="utf-8")
    )
    assert job.provide is None


def test_a_dispatch_builds_the_addressed_environment_before_the_wave_starts(
    dispatcher: Dispatcher, backend: RecordingScheduler, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One process, before anything is queued, so a wave of nodes finds its environment built.

    It runs from the mirror, where a host keeps its built environments, and builds from the
    snapshot's own copy of the artifact, so what a job activates and what was built from are
    the same content.
    """
    del backend
    machine = machine_with()
    monkeypatch.setattr(dispatch_module, "connection", lambda host: machine)
    announced: list[str] = []
    dispatcher.run(
        plan(),
        shipped(dispatcher, "python -m foo"),
        root="/repo",
        resources=Resources(),
        watch=announced.append,
        prefix="/repo/.mainboard/prefixes/default/abcd1234",
    )

    [asked] = [line for line in machine.lines if "provide" in line]
    assert asked.startswith("cd /repo && ")
    assert "mainboard provide default --source /repo/.mainboard/dispatch/sources/" in asked
    assert asked.endswith(" --expect abcd1234 >/dev/null")
    assert [told for told in announced if told.startswith("built default on gold for /repo")]


def test_a_failed_prefix_build_prevents_scheduler_submission(
    dispatcher: Dispatcher, backend: RecordingScheduler, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A prior workspace check cannot prove that a newly addressed prefix builds."""
    machine = machine_with(rules=[("provide", 1, "no pixi here")])
    monkeypatch.setattr(dispatch_module, "connection", lambda host: machine)
    with pytest.raises(SystemExit, match="could not build default on gold: no pixi here"):
        dispatcher.run(
            plan(),
            shipped(dispatcher, "python -m foo"),
            root="/repo",
            resources=Resources(),
            prefix="/repo/.mainboard/prefixes/default/abcd1234",
        )

    assert not any(name == "submit" for name, _ in backend.calls)


def test_source_identity_is_read_once_for_script_and_snapshot(
    dispatcher: Dispatcher,
    backend: RecordingScheduler,
    workdir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(dispatch_module, "connection", lambda host: machine_with())
    shipment = shipped(dispatcher, "python -m foo")
    (workdir / "mainboard.toml").write_text("# later edit")
    dispatcher.run(plan(), shipment, root="/repo", resources=Resources())
    [(root, script, _)] = [call for name, call in backend.calls if name == "submit"]
    assert root.endswith(shipment.source.key)
    assert str(root) in (workdir / ".mainboard/dispatch/jobs" / Path(script).name).read_text()


def test_equal_source_reuses_snapshot_and_changed_bytes_get_a_new_one(
    dispatcher: Dispatcher,
    workdir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PATH", "")
    first = dispatcher.pinned("/repo", source=dispatcher.source())
    assert "/sources/sha256-" in first
    assert dispatcher.pinned("/repo", source=dispatcher.source()) == first
    (workdir / "mainboard.toml").write_text("# changed")
    assert dispatcher.pinned("/repo", source=dispatcher.source()) != first


def test_submit_refuses_a_broken_environment_and_names_the_host_a_scheduler_rejected(
    dispatcher: Dispatcher, backend: RecordingScheduler, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A broken remote env becomes one plain sentence instead of a traceback inside a job log."""

    def reject(
        remote: Machine, root: str, *, script: str, args: Sequence[str], resources: Resources
    ) -> str:
        raise SystemExit("no PBS queue resolved")

    backend.submit = reject
    with pytest.raises(SystemExit, match="submission to host 'gold' failed"):
        dispatcher.submit(plan(), "/repo", script="train.sh", args=(), resources=Resources())
    monkeypatch.setattr(
        dispatch_module,
        "connection",
        lambda host: machine_with(rules=[("true", 1, "ModuleNotFoundError: no torch")]),
    )
    with pytest.raises(SystemExit, match="environment on 'gold' is broken: ModuleNotFoundError"):
        dispatcher.submit(plan(), "/repo", script="train.sh", args=(), resources=Resources())
    # A host whose tool predates the job runner would queue the job and lose it at start.
    monkeypatch.setattr(
        dispatch_module,
        "connection",
        lambda host: machine_with(rules=[("job --help", 1, 'Error: Unknown command "job"')]),
    )
    with pytest.raises(SystemExit, match=r"cannot run a job .*`mainboard setup gold`"):
        dispatcher.submit(plan(), "/repo", script="train.sh", args=(), resources=Resources())


def test_await_many_polls_every_handle_until_terminal_and_persists_what_it_learned(
    dispatcher: Dispatcher, backend: RecordingScheduler
) -> None:
    """A transient blip is not a verdict, so that handle is simply retried on the next tick."""
    assert inspect.signature(Dispatcher.await_many).parameters["interval"].default == POLL_SECONDS
    probes = {"n": 0}
    states = [
        HostUnreachable("blip"),
        JobState(handle="H1", state="R", verdict="running"),
        JobState(handle="H1", state="F", exit_code=0, verdict="ok"),
    ]

    def answer(remote: Machine, root: str, *, handle: str) -> JobState:
        probes["n"] += 1
        reply = states[min(probes["n"] - 1, len(states) - 1)]
        if isinstance(reply, HostUnreachable):
            raise reply
        return reply

    backend.state = answer
    dispatcher.cache.record(run_record("H1"))
    settled = Handle(id="H1", host="gold", root="/repo", kind="ssh")
    assert dispatcher.await_many([settled])[settled].ok
    assert probes["n"] == 3
    assert dispatcher.cache.run("H1").verdict == "ok"
    del backend.state
    backend.state_result = JobState(handle="H2", state="F", exit_code=137, verdict="failed")
    unrecorded = Handle(id="H2", host="gold", root="/repo", kind="ssh")
    verdict = dispatcher.await_many([unrecorded])[unrecorded]
    assert (verdict.verdict, verdict.exit_code) == ("failed", 137)
    assert "memory" in verdict.reason


def test_probe_absorbs_an_unreachable_host_while_state_names_it(
    dispatcher: Dispatcher, backend: RecordingScheduler
) -> None:
    """A caller polling on its own cadence must not record a state the host never reported."""

    def down(remote: Machine, root: str, *, handle: str) -> JobState:
        raise HostUnreachable("ssh connect to 'gold' failed: connection timed out")

    backend.state = down
    handle = Handle(id="H1", host="gold", root="/repo", kind="ssh")
    assert dispatcher.probe(handle) is None
    with pytest.raises(HostUnreachable, match="timed out"):
        dispatcher.state(handle)


def test_states_asks_the_host_once_and_only_re_asks_what_the_listing_missed(
    dispatcher: Dispatcher, backend: RecordingScheduler
) -> None:
    """One listing for the whole host, then one further question per handle it did not cover.

    A dispatch cache that has been accumulating for months holds a thousand runs on one box, so
    the listing is what keeps a sweep to a single round trip. What the listing does not span (a
    `squeue` that only sees live jobs, a PBS server that purged its history) is where the job's
    real ending is, and that is worth one question each rather than a guess.
    """
    backend.state_result = JobState(handle="H1", state="F", exit_code=0, verdict="ok")
    listed = [Handle(id=name, host="gold", root="/repo", kind="ssh") for name in ("H1", "H2")]
    resolved = dispatcher.states([*listed, listed[0]])
    assert sorted(resolved) == ["H1", "H2"]
    assert backend.calls == [("states", ("/repo", ("H1", "H2"))), ("state", ("/repo", "H2"))]
    assert dispatcher.states([]) == {}


@pytest.mark.parametrize(
    "fetch_path",
    ["out/", "a/b/c.json"],
)
def test_fetch_pulls_the_recorded_path_back_into_its_own_parent_directory(
    dispatcher: Dispatcher,
    workdir: Path,
    monkeypatch: pytest.MonkeyPatch,
    fetch_path: str,
) -> None:
    """The results land under the workspace, wherever the command that pulls them was typed."""
    (workdir / "mainboard.toml").write_text(
        "[workspace]\nname = 'test'\n[hosts.gold]\npython = 'remote-python'\n"
    )
    pulled = []
    monkeypatch.setattr(
        dispatch_module.Collector,
        "pull",
        lambda self, host, **kwargs: pulled.append((self.root, host, kwargs)),
    )
    dispatcher.fetch(Handle(id="H1", host="gold", root="/repo", kind="ssh", fetch_path=fetch_path))
    assert pulled == [
        (
            workdir,
            "gold",
            {
                "root": "/repo",
                "path": fetch_path.rstrip("/"),
                "python": "remote-python",
            },
        )
    ]
    with pytest.raises(LookupError, match="no fetch path"):
        dispatcher.fetch(Handle(id="H1", host="gold", root="/repo", kind="ssh"))


def test_a_collection_that_could_not_be_trusted_is_refused_naming_the_path_and_host(
    dispatcher: Dispatcher, workdir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A conflict, a failed stream or a torn archive all end the fetch with one mission error."""
    (workdir / "mainboard.toml").write_text("[workspace]\nname = 'test'\n[hosts.gold]\n")

    def conflicting(self: dispatch_module.Collector, host: str, **_: str) -> int:
        raise ValueError("conflicting collected evidence, local copy preserved: out/a")

    monkeypatch.setattr(dispatch_module.Collector, "pull", conflicting)
    with pytest.raises(MissionError, match="collection of out/ from gold failed: conflicting"):
        dispatcher.fetch_path("gold", root="/repo", path="out/")


def test_a_rendered_and_a_staged_script_are_both_content_addressed(
    dispatcher: Dispatcher, workdir: Path
) -> None:
    """Repeated runs reuse the file instead of growing the jobs directory unboundedly."""
    spec = JobSpec(cmd="python -m foo", plan=plan(), root="/repo")
    rendered = dispatcher.write_job_script(spec, pbs=False)
    assert rendered.startswith(".mainboard/dispatch/jobs/")
    assert dispatcher.write_job_script(spec, pbs=False) == rendered
    assert dispatcher._prepare_script("job") == ("job", ())  # ruff:ignore[private-member-access]  reason=unit-tests the module-private staging helper since=2026-08-16
    external = workdir.parent / "external.sh"
    external.write_text("#!/bin/bash\necho hi\n")
    prepared, staged = dispatcher._prepare_script(str(external))  # ruff:ignore[private-member-access]  reason=unit-tests the module-private staging helper since=2026-08-16
    assert staged == (prepared,)
    assert (workdir / prepared).read_text() == external.read_text()
    with pytest.raises(FileNotFoundError, match="cannot be shipped to the host"):
        dispatcher._prepare_script("./missing/job.sh")  # ruff:ignore[private-member-access]  reason=unit-tests the module-private staging helper since=2026-08-16


@pytest.mark.parametrize("local_output", ["missing", "empty", "stale"])
def test_concurrent_mirrors_preserve_declared_outputs_and_prune_source(
    workdir: Path, monkeypatch: pytest.MonkeyPatch, local_output: str
) -> None:
    """Missing or empty local data cannot erase another running job's declared output."""
    if which("rsync") is None:
        pytest.skip("the optional rsync executable is not installed")
    source, landed = workdir / "project", workdir / "host-side"
    source.mkdir()
    (source / "current.py").write_text("current source")
    (source / "selected-input.json").write_text("deliberately bound input")
    output = "project/measurements[trial]"
    if local_output != "missing":
        (workdir / output).mkdir()
    if local_output == "stale":
        (workdir / output / "plain.jsonl").write_text("stale local output must never upload\n")
        (workdir / output / "local-only.json").write_text("not an input")
    remote = landed / output
    remote.mkdir(parents=True)
    (landed / "project/stale.py").write_text("retired source")
    (landed / "project/measurementst").write_text("not the literal output path")
    (landed / "project/new-output").mkdir()
    database = workdir / "state.sqlite"
    store = Cache(database)
    store.record(
        run_record("old", target="other-alias").model_copy(
            update={"fetch_path": output, "verdict": "ok"}
        )
    )
    real = dispatch_module.rsync
    monkeypatch.setattr(
        dispatch_module,
        "rsync",
        lambda sources, dest, flags, **k: real(sources, f"{landed}/", flags, **k),
    )
    ready = Barrier(2)

    def mirror(job: int) -> None:
        instance = Dispatcher(cache=Cache(database), sync=GitignoreFilter(workdir))
        host = plan(profile=HostProfile(kind="ssh", root="/repo", sync={"include": ["project"]}))
        with closing(instance.cache.connection):
            ready.wait(timeout=10)
            instance.rsync_up(
                host, "/repo", fetch="project/new-output", extra=["project/selected-input.json"]
            )
            instance.cache.record(
                run_record(str(job)).model_copy(update={"fetch_path": "project/new-output"})
            )
        del instance

    artifact = remote / "plain.jsonl"
    with artifact.open("w") as writer, ThreadPoolExecutor(max_workers=2) as workers:
        writer.write("before\n")
        writer.flush()
        list(workers.map(mirror, (1, 2)))
        writer.write("after\n")
    assert artifact.read_text() == "before\nafter\n"
    assert not (remote / "local-only.json").exists()
    assert (landed / "project/new-output").is_dir()
    assert (landed / "project/current.py").read_text() == "current source"
    assert (landed / "project/selected-input.json").read_text() == "deliberately bound input"
    assert not (landed / "project/stale.py").exists()
    assert not (landed / "project/measurementst").exists()


@pytest.mark.parametrize("recorded", [False, True])
@pytest.mark.parametrize("resource", ["project", "project/output", "project/output/row.json"])
def test_explicit_output_resource_is_refused_before_transfer(
    workdir: Path, monkeypatch: pytest.MonkeyPatch, recorded: bool, resource: str
) -> None:
    source = workdir / "project/output"
    source.mkdir(parents=True)
    (source / "row.json").write_text("stale local data")
    instance = Dispatcher(cache=cache(), sync=GitignoreFilter(workdir))
    if recorded:
        instance.cache.record(
            run_record("old").model_copy(update={"fetch_path": "project/output"})
        )
    monkeypatch.setattr(
        dispatch_module, "rsync", lambda *a, **kw: pytest.fail("transport reached")
    )
    execution = plan(profile=HostProfile(kind="ssh", root="/repo", sync={"include": ["project"]}))
    with pytest.raises(ValueError, match="separate immutable input"):
        instance.rsync_up(
            execution,
            "/repo",
            extra=[resource],
            fetch="" if recorded else "project/output",
        )


def test_submission_records_outputs_before_releasing_the_mirror_lock(
    dispatcher: Dispatcher, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The next submission cannot miss the previous handle's output declaration."""
    execution = plan()
    record = dispatcher.cache.record

    def within_transaction(run) -> None:
        assert SyncLock(execution.host, dispatcher.root).lock.is_locked
        record(run)

    monkeypatch.setattr(dispatcher.cache, "record", within_transaction)
    handle = dispatcher.submit(
        execution,
        "/repo",
        script="train.sh",
        args=(),
        resources=Resources(),
        fetch="project/output",
    )
    assert dispatcher.cache.run(handle).fetch_path == "project/output"


@pytest.mark.parametrize("path", ["/absolute", "../escape", "."])
def test_mirror_refuses_unsafe_declared_output_protection(workdir: Path, path: str) -> None:
    instance = Dispatcher(cache=cache(), sync=GitignoreFilter(workdir))
    with pytest.raises(ValueError, match="relative path below"):
        instance._protected_outputs(path)


def test_a_windows_host_is_mirrored_by_tarball_under_the_rules_rsync_would_have_run(
    workdir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No rsync answers on the far side, so the same file set is handed to the tar route whole."""
    (workdir / "src").mkdir()
    mirrored: list[tuple[str, str, dict[str, Sequence[str] | str]]] = []

    class Recording:
        def __init__(self, root: Path, ssh: SshTransport) -> None:
            del root, ssh

        def mirror(self, execution: ExecutionPlan, root: str, **rules: Sequence[str]) -> None:
            mirrored.append((execution.host, root, rules))

    monkeypatch.setattr(dispatch_module, "Tarball", Recording)
    monkeypatch.setattr(dispatch_module, "rsync", lambda *a, **k: pytest.fail("rsync reached"))
    instance = Dispatcher(cache=cache(), sync=GitignoreFilter(workdir))
    instance.cache.save_host(HostSetup(host="homelab", root="C:/w"))
    profile = HostProfile(kind="ssh", root="C:/w", platform="win-64", sync={"include": ["src"]})
    assert instance.rsync_up(plan(host="homelab", profile=profile), "C:/w") == ["src"]
    [(host, root, rules)] = mirrored
    assert (host, root, rules["paths"][0], rules["vendored"]) == ("homelab", "C:/w", "src", "")
    assert "/src/***" not in rules["exclude"], "nothing required, so no remainder rule"
    assert instance.cache.host("homelab").synced_at


def test_a_second_request_while_a_creation_is_unresolved_is_refused_before_any_provider(
    workdir: Path,
) -> None:
    """The first may have allocated a billable instance, so its label is reconciled first."""
    instance = Dispatcher(cache=cache(), sync=GitignoreFilter(workdir))
    shipment = shipped(instance, "python -m train")
    instance.allocating(plan(), shipment, Resources(), evidence="not_started").begin()
    with pytest.raises(MissionError, match="unresolved creation"):
        instance.allocating(plan(), shipment, Resources(), evidence="not_started")


def test_rsync_up_refuses_an_undeclared_include_and_warns_about_a_stale_one(
    workdir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    instance = Dispatcher(root=workdir)
    empty = HostProfile(kind="ssh", root="/repo", sync={"include": []})
    with pytest.raises(LookupError, match="nothing to sync"):
        instance.rsync_up(plan(profile=empty), "/repo")
    gone = HostProfile(kind="ssh", root="/repo", sync={"include": ["gone"]})
    with pytest.raises(LookupError, match="missing locally"):
        instance.rsync_up(plan(profile=gone), "/repo")
    (workdir / "src").mkdir()
    partly = HostProfile(kind="ssh", root="/repo", sync={"include": ["src", "packages/meteng"]})
    sent: dict[str, str | Sequence[str]] = {}
    monkeypatch.setattr(
        dispatch_module, "rsync", lambda sources, dest, flags, **k: sent.update(sources=sources)
    )
    warned: list[tuple[str, tuple[int | str, ...]]] = []
    monkeypatch.setattr(dispatch_module.logger, "warning", lambda msg, *a: warned.append((msg, a)))
    instance.rsync_up(plan(profile=partly), "/repo")
    assert sent["sources"][0] == "src"
    [(message, args)] = warned
    assert "stale sync include" in message
    assert args[:2] == (1, "packages/meteng")


def test_rsync_up_punches_a_required_group_through_the_denylist_or_refuses_an_incomplete_one(
    workdir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The compiled artifact must ride the mirror whole, since a half lock installs nothing."""
    (workdir / "src").mkdir()
    envdir = workdir / ".mainboard/envs/default"
    envdir.mkdir(parents=True)
    (envdir / "pixi.toml").write_text("x")
    group = (".mainboard/envs/default/pixi.toml", ".mainboard/envs/default/pixi.lock")
    instance = Dispatcher(cache=cache(), sync=GitignoreFilter(workdir))
    host = plan(profile=HostProfile(kind="ssh", root="/repo", sync={"include": ["src"]}))
    with pytest.raises(LookupError, match="incomplete"):
        instance.rsync_up(host, "/repo", required=[group])
    (envdir / "pixi.lock").write_text("y")
    sent: dict[str, str | Sequence[str] | bool] = {}
    monkeypatch.setattr(
        dispatch_module,
        "rsync",
        lambda sources, dest, flags, **k: sent.update(sources=sources, **k),
    )
    instance.rsync_up(host, "/repo", required=[group])
    assert ".mainboard/envs/default/pixi.toml" in sent["sources"]
    assert "/.mainboard/" in sent["include"]
    assert "/.mainboard/***" in sent["exclude"]
    assert sent["allow_vanished"] is False


@pytest.mark.parametrize("spelling", ("required", "extra", "include"))
@pytest.mark.parametrize("resource", ("src/.card.lock.local.0", "src/.card.lock.local/input.json"))
def test_rsync_up_refuses_explicit_card_lease_resources_before_transfer(
    workdir: Path,
    monkeypatch: pytest.MonkeyPatch,
    spelling: str,
    resource: str,
) -> None:
    path = workdir / resource
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("not transferable")
    instance = Dispatcher(cache=cache(), sync=GitignoreFilter(workdir))
    include = [resource] if spelling == "include" else ["src"]
    host = plan(profile=HostProfile(kind="ssh", root="/repo", sync={"include": include}))
    monkeypatch.setattr(
        dispatch_module, "rsync", lambda *_args, **_kwargs: pytest.fail("transfer reached")
    )
    with pytest.raises(ValueError, match="card leases cannot be declared"):
        instance.rsync_up(
            host,
            "/repo",
            required=[(resource,)] if spelling == "required" else (),
            extra=(resource,) if spelling == "extra" else (),
        )


def test_a_narrow_host_mirrors_named_job_files_without_touching_other_projects(
    workdir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The pruning under test is what a remote mirror does, which only upstream rsync performs;
    # Apple's openrsync, the only rsync on a stock macOS runner, deletes the excluded paths too.
    try:
        binary(mirror=True)
    except RuntimeError:
        pytest.skip("mirroring needs upstream rsync, and only Apple's openrsync is installed")
    job = "research/camp/experiments/node/run.py"
    untouched = (
        "research/camp/papers/frozen.tex",
        "research/other/source.py",
        "packages/unrelated/src/module.py",
    )
    landed = workdir / "host-side"
    for path in (job, *untouched):
        (workdir / path).parent.mkdir(parents=True, exist_ok=True)
        (workdir / path).write_text("local\n")
        (landed / path).parent.mkdir(parents=True, exist_ok=True)
        (landed / path).write_text("remote\n")
    (workdir / "mainboard.toml").write_text("[workspace]\nname = 'lab'\n")
    real = dispatch_module.rsync
    monkeypatch.setattr(
        dispatch_module,
        "rsync",
        lambda sources, dest, flags, **k: real(sources, f"{landed}/", flags, **k),
    )
    base = HostProfile(sync={"include": ["research", "packages"]})
    profile = HostProfile(sync={"include": ["mainboard.toml"]}).inheriting(base)
    instance = Dispatcher(cache=cache(), sync=GitignoreFilter(workdir))
    instance.rsync_up(plan(profile=profile), "/repo", extra=[job])
    assert (landed / job).read_text() == "local\n"
    assert (landed / "mainboard.toml").is_file()
    assert all((landed / path).read_text() == "remote\n" for path in untouched)


def test_a_real_mirror_carries_the_staged_job_script_past_a_required_group(
    workdir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The one end-to-end proof that a dispatch shipping both actually delivers both.

    A required group is punched through the denylist by naming its own directory and then
    excluding everything under it that was not asked for, and the staged job script lives under
    that same generated directory. Named only as a source it is dropped by that remainder rule,
    which is how a landed rental was told to run a script the mirror never carried and answered
    `No such file or directory`, exit 127 (vast 49865738, 2026-09-04). Nothing but a real
    transfer can tell the two filter sets apart, so this runs one.
    """
    if which("rsync") is None:
        pytest.skip("the optional rsync executable is not installed")
    (workdir / "src").mkdir()
    (workdir / "src/run.py").write_text("print(1)")
    envdir = workdir / ".mainboard/envs/default"
    envdir.mkdir(parents=True)
    (envdir / "pixi.toml").write_text("x")
    (envdir / "pixi.lock").write_text("y")
    jobs = workdir / ".mainboard/dispatch/jobs"
    jobs.mkdir(parents=True)
    (jobs / "job-abc.sh").write_text("#!/bin/bash\nexit 0\n")
    landed = workdir / "host-side"
    real = dispatch_module.rsync
    monkeypatch.setattr(
        dispatch_module,
        "rsync",
        lambda sources, dest, flags, **k: real(sources, f"{landed}/", flags, **k),
    )
    instance = Dispatcher(cache=cache(), sync=GitignoreFilter(workdir))
    host = plan(profile=HostProfile(kind="ssh", root="/repo", sync={"include": ["src"]}))
    group = (".mainboard/envs/default/pixi.toml", ".mainboard/envs/default/pixi.lock")
    script = ".mainboard/dispatch/jobs/job-abc.sh"
    instance.rsync_up(host, "/repo", required=[group], extra=[script])
    assert (landed / "src/run.py").is_file()
    assert (landed / group[0]).is_file()
    assert (landed / script).is_file()


def test_a_real_mirror_carries_a_vendored_dependency_as_the_files_its_links_refer_to(
    workdir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A house package outside the workspace root is compiled at `.mainboard/vendor/<dist>`.

    On the machine that has the source that directory holds links into it, which is what keeps
    the editable install editable. A link is the one thing this transfer must not carry: landed
    as a link on a host it points at a tree nothing ever put there, and every environment built
    from the shipped lock dies on a path that is not a Python project. Only a real transfer can
    say which of the two arrived.
    """
    if which("rsync") is None:
        pytest.skip("the optional rsync executable is not installed")
    (workdir / "src").mkdir()
    (workdir / "src/run.py").write_text("print(1)")
    envdir = workdir / ".mainboard/envs/default"
    envdir.mkdir(parents=True)
    (envdir / "pixi.toml").write_text("x")
    (envdir / "pixi.lock").write_text("y")
    source = workdir / "outside/sample_lib"
    (source / "src/sample_lib").mkdir(parents=True)
    (source / "src/sample_lib/__init__.py").write_text("SHADE = 'ai'\n")
    (source / "pyproject.toml").write_text('[project]\nname = "sample-lib"\n')
    vendored = workdir / ".mainboard/vendor/sample-lib"
    vendored.mkdir(parents=True)
    for entry in sorted(source.iterdir()):
        (vendored / entry.name).symlink_to(entry)
    (source / "src/sample_lib/__pycache__").mkdir()
    (source / "src/sample_lib/__pycache__/stale.pyc").write_text("noise")
    landed = workdir / "host-side"
    real = dispatch_module.rsync
    monkeypatch.setattr(
        dispatch_module,
        "rsync",
        lambda sources, dest, flags, **k: real(sources, f"{landed}/", flags, **k),
    )
    instance = Dispatcher(cache=cache(), sync=GitignoreFilter(workdir))
    host = plan(profile=HostProfile(kind="ssh", root="/repo", sync={"include": ["src"]}))
    group = (".mainboard/envs/default/pixi.toml", ".mainboard/envs/default/pixi.lock")

    instance.rsync_up(host, "/repo", required=[group])

    arrived = landed / ".mainboard/vendor/sample-lib"
    assert arrived.is_dir() and not arrived.is_symlink()
    assert not (arrived / "src").is_symlink()
    assert (arrived / "src/sample_lib/__init__.py").read_text() == "SHADE = 'ai'\n"
    assert (arrived / "pyproject.toml").is_file()
    assert not (arrived / "src/sample_lib/__pycache__").exists()


def test_a_workspace_with_nothing_vendored_ships_no_second_transfer(
    workdir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The rule costs a workspace that declares no outside dependency nothing at all."""
    (workdir / "src").mkdir()
    (workdir / "src/run.py").write_text("print(1)")
    sent: list[Sequence[str]] = []
    monkeypatch.setattr(
        dispatch_module, "rsync", lambda sources, dest, flags, **k: sent.append(sources) or ""
    )
    instance = Dispatcher(cache=cache(), sync=GitignoreFilter(workdir))
    host = plan(profile=HostProfile(kind="ssh", root="/repo", sync={"include": ["src"]}))

    instance.rsync_up(host, "/repo")

    assert len(sent) == 1


@pins_on_this_host
@setgid_inherits
def test_a_real_mirror_never_overrides_the_hosts_setgid_group(
    workdir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`.mainboard/envs/<env>`, punched through the denylist, must inherit the host's group.

    A directory made under a setgid parent inherits that parent's group and its own setgid bit
    for free; `ARCHIVE`'s `-p`/`-g`/`-o` used to overwrite the first implied directory this
    transfer creates with the workstation's own mode and group instead, undoing that inheritance
    the instant it happened. An 8 GB prefix landed on the invoking user's personal group rather
    than the shared project one this way and blew its inode quota (Miyabi, 2026-09-05).

    Needs a second group the runner belongs to, to stand in for the shared project group a
    setgid mirror root carries: skipped where there is only one to pick from.
    """
    if which("rsync") is None:
        pytest.skip("the optional rsync executable is not installed")
    groups = sorted({os.getgid(), *os.getgroups()})
    if len(groups) < 2:
        pytest.skip("the runner belongs to a single group, so no group mismatch can be shown")
    project_gid = next(gid for gid in groups if gid != os.getgid())
    (workdir / "src").mkdir()
    (workdir / "src/run.py").write_text("print(1)")
    envdir = workdir / ".mainboard/envs/default"
    envdir.mkdir(parents=True)
    (envdir / "pixi.toml").write_text("x")
    (envdir / "pixi.lock").write_text("y")
    landed = workdir / "host-side"
    landed.mkdir()
    os.chown(landed, -1, project_gid)
    landed.chmod(landed.stat().st_mode | stat.S_ISGID)
    real = dispatch_module.rsync
    monkeypatch.setattr(
        dispatch_module,
        "rsync",
        lambda sources, dest, flags, **k: real(sources, f"{landed}/", flags, **k),
    )
    instance = Dispatcher(cache=cache(), sync=GitignoreFilter(workdir))
    host = plan(profile=HostProfile(kind="ssh", root="/repo", sync={"include": ["src"]}))
    group = (".mainboard/envs/default/pixi.toml", ".mainboard/envs/default/pixi.lock")
    instance.rsync_up(host, "/repo", required=[group])
    for made in (
        landed / ".mainboard",
        landed / ".mainboard/envs",
        landed / ".mainboard/envs/default",
    ):
        found = made.stat()
        assert found.st_gid == project_gid, (
            f"{made} landed on group {found.st_gid}, not {project_gid}"
        )
        assert found.st_mode & stat.S_ISGID, f"{made} lost its inherited setgid bit"


@pytest.mark.parametrize(
    ("extra", "raised", "detail"),
    [
        ((), ProcessExecutionError, "exit code: 23"),
        ((".mainboard/dispatch/jobs/job-x.sh",), RuntimeError, "submission aborted"),
    ],
)
def test_rsync_up_preserves_an_ordinary_mirror_error_but_wraps_a_failed_required_transfer(
    workdir: Path,
    monkeypatch: pytest.MonkeyPatch,
    extra: tuple[str, ...],
    raised: type[BaseException],
    detail: str,
) -> None:
    (workdir / "src").mkdir()
    instance = Dispatcher(cache=cache(), sync=GitignoreFilter(workdir))

    def fail(*a, **k) -> None:
        raise ProcessExecutionError(["rsync"], 23, "", "connection reset")

    monkeypatch.setattr(dispatch_module, "rsync", fail)
    host = plan(profile=HostProfile(kind="ssh", root="/repo", sync={"include": ["src"]}))
    with pytest.raises(raised, match=detail):
        instance.rsync_up(host, "/repo", extra=extra)


def test_the_manifest_owns_the_walltime_default_never_the_dispatch_code() -> None:
    """The `00:30:00` fallback lives on the manifest schema's `Defaults`, never in dispatch."""
    assert Defaults().walltime == "00:30:00"
    source = Path(dispatch_module.__file__).read_text(encoding="utf-8")
    assert "debug-g" not in source
    assert "00:30:00" not in source


def test_a_held_dispatch_is_one_durable_row_carrying_the_request_that_makes_it_again(
    dispatcher: Dispatcher,
) -> None:
    """A request held only in the process that made it dies with that process.

    So it goes into the same registry every dispatched run lives in, keyed by an id of our own
    since no scheduler ever took it, and holding the same job twice keeps one row rather than
    growing one per attempt. The row carries the ask itself, which is what lets a sweep on
    another day make exactly the same dispatch.
    """
    asked = Request(
        target="miyabi-g",
        command="python -m foo",
        name="batch:wave/shell",
        queue="short-g",
        walltime="06:00:00",
        gpus=1,
    )
    held = dispatcher.hold(asked, reason="would exceed group xg25g007's limit on resource njobs-g")
    assert held.handle.startswith("held-")
    assert (held.verdict, held.state) == ("held", "held")
    assert held.request == asked and "njobs-g" in held.reason
    again = dispatcher.hold(asked, reason="still no room")
    assert (again.handle, again.submitted_at) == (held.handle, held.submitted_at)
    assert [run.handle for run in dispatcher.cache.live()] == [held.handle]
    # The row is a placeholder for a request, so it leaves the table when the request goes
    # through rather than settling into a verdict the job never had.
    dispatcher.cache.forget(again)
    assert dispatcher.cache.live() == []


def test_a_dispatch_ships_the_very_artifact_it_addressed_its_environment_by(
    dispatcher: Dispatcher, backend: RecordingScheduler, workdir: Path
) -> None:
    """A submit used to pin an address and ship no artifact for anyone to reach it by.

    The compiled pair lives under the generated tree, which every mirror denies, so only
    `setup`, `sync` and a rental landing named it and only they carried it. A submit therefore
    addressed the workstation's compile and left the host holding whatever its own last compile
    had produced. On 2026-09-06 two task rows were added to the monorepo manifest and the
    mirror's compile stayed at the morning's, missing `[tasks.head-paper]`: the workstation
    pinned a9d234f5f0dd2e93, the host read db8171ec0bd191b2, and every job of the wave died at
    environment prime with a mirror nobody had told to catch up.
    """
    trio = (
        ".mainboard/envs/default/pixi.toml",
        ".mainboard/envs/default/pixi.lock",
        ".mainboard/envs/default/state.toml",
    )

    dispatcher.run(
        plan(),
        shipped(dispatcher, "python -m foo"),
        root="/repo",
        resources=Resources(),
        artifact=trio,
    )

    del backend, workdir
    assert dispatcher.required == [[list(trio)]]
