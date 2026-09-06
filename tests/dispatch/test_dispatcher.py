import inspect
import os
import stat
from collections.abc import Callable, Sequence
from pathlib import Path
from shutil import which
from typing import TYPE_CHECKING

import pytest
from plumbum.commands.processes import ProcessExecutionError

from mainboard import ExecutionPlan, MissionError
from mainboard.dispatch import Dispatcher, GitignoreFilter, Handle, HostSetup, Verdict, shared
from mainboard.dispatch import dispatcher as dispatch_module
from mainboard.dispatch import provenance as provenance_module
from mainboard.dispatch.jobs import JobSpec
from mainboard.dispatch.schedulers import HostUnreachable, registry
from mainboard.dispatch.shipment import Shipment
from mainboard.dispatch.vocabulary import POLL_SECONDS, JobState, Request, Resources
from mainboard.manifest import Container, Defaults, HostProfile, QueuePolicy

from .support import (
    RecordingMachine,
    RecordingScheduler,
    cache,
    machine_with,
    plan,
    run_record,
)

if TYPE_CHECKING:
    from mainboard.dispatch.transport import Machine

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


def answering(**told: str) -> Callable[..., str]:
    """A `git` that answers each question by the flag or verb naming it, silence otherwise.

    `told` keys are the distinguishing tokens (`toplevel`, `describe`, `HEAD`, `ls-files`,
    `status`, `diff`), so a test says what a tree looks like and nothing else.
    """

    def git(*args: str) -> str:
        if "--show-toplevel" in args:
            return told.get("toplevel", "/repo")
        if "describe" in args:
            return told.get("describe", "")
        if "HEAD" in args and "diff" not in args:
            return told.get("HEAD", "abc1234")
        if "ls-files" in args:
            return told.get("ls_files", "")
        if "status" in args:
            return told.get("status", "")
        return told.get("diff", "")

    return git


def shipped(dispatcher: Dispatcher, command: str, imports: tuple[str, ...] = ()) -> Shipment:
    """`command` as a board ships it to the dispatcher: the mirror, under the tree's provenance."""
    return Shipment.of_command(command, source=dispatcher.source(command), imports=imports)


@pytest.fixture
def backend(monkeypatch: pytest.MonkeyPatch) -> RecordingScheduler:
    """Pin the backend, the connection, git and the clock, every seam a dispatch reaches."""
    scheduler = RecordingScheduler()
    monkeypatch.setattr(dispatch_module, "pick", lambda profile: scheduler)
    monkeypatch.setattr(registry, "SCHEDULERS", _StubStrategy(scheduler))
    monkeypatch.setattr(dispatch_module, "connection", lambda host: machine_with())
    monkeypatch.setattr(
        dispatch_module, "git", lambda *args: "abc1234" if args[0] == "rev-parse" else ""
    )
    monkeypatch.setattr(provenance_module, "git", answering())
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
    assert script.startswith(".mainboard/dispatch/jobs/")
    assert args == ()
    assert dispatcher.shipped == [(script,)]
    assert (workdir / script).is_file()
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
    text = generated.read_text()
    # The script activates through the snapshot this dispatch pinned, whose `.mainboard` is a
    # symlink back to the mirror, so the job gets the mirror's environment out of a tree whose
    # code no later sync can rewrite.
    pinned = dispatcher.pinned("/repo", source=dispatcher.source())
    assert pinned.startswith("/repo/.mainboard/dispatch/sources/")
    assert f"{pinned}/.mainboard/activate-serving.sh" in text
    assert f"{pinned}/.mainboard/envs/serving/.pixi/envs/serving/bin" in text


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
    text = (workdir / script).read_text()
    assert "apptainer exec image.sif bash -c 'python -m foo' || status=$?" in text


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


@pytest.mark.parametrize(("porcelain", "dirty"), [("", 0), ("M x.py", 1)])
def test_submit_records_the_run_with_the_git_provenance_it_was_dispatched_from(
    dispatcher: Dispatcher,
    backend: RecordingScheduler,
    monkeypatch: pytest.MonkeyPatch,
    porcelain: str,
    dirty: int,
) -> None:
    monkeypatch.setattr(
        provenance_module,
        "git",
        answering(describe="abc1234", ls_files="100644 aaaa 0\ta.py", status=porcelain),
    )
    handle = dispatcher.submit(
        plan(), "/repo", script="train.sh", args=("--x", "1"), resources=Resources()
    )
    [run] = dispatcher.cache.recent(10)
    assert (run.handle, run.target, run.git_sha, run.dirty) == (handle, "gold", "abc1234", dirty)
    assert run.args == "--x 1"
    # And the whole commit beside the short one, with the digest of the tree it was taken from:
    # the mirror this job runs in has no history, so the registry is where a later reader learns
    # what the run was measured from.
    assert run.commit == "abc1234"
    assert len(run.digest) == 64


def test_a_dispatch_runs_from_a_snapshot_of_the_mirror_and_never_from_the_mirror_itself(
    dispatcher: Dispatcher, backend: RecordingScheduler, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The fault this exists for: a later sync of another tree rewrote the code under live jobs."""
    machine = machine_with()
    monkeypatch.setattr(dispatch_module, "connection", lambda host: machine)
    monkeypatch.setattr(
        dispatch_module, "git", lambda *args: "abc1234" if args[0] == "rev-parse" else ""
    )
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
    assert f"mb_snap={pinned}" in built
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
    body = (workdir / str(script)).read_text(encoding="utf-8")
    assert f"cd /repo && mainboard provide default --source {root}/.mainboard/envs/default" in body
    # And it names the address the dispatch pinned, so a host that reads the shipped artifact as
    # another environment says which two addresses and which two pixis instead of building one
    # beside the one every job of the wave is waiting for.
    assert "--expect abcd1234" in body
    assert f"source {prefix}/activate.sh" in body
    assert f"export LD_LIBRARY_PATH={prefix}/.pixi/envs/default/lib${{LD_LIBRARY_PATH:+" in body
    assert body.index("provide default") < body.index("activate.sh")
    # Nothing in the job asks pixi to reconcile anything, which is what made a shared prefix a
    # race in the first place.
    assert "run --env" not in body
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
            imports=("src", "packages/lab-core/src", ".mainboard/vendor/paleta-tsukuba/src"),
        ),
        root="/repo",
        resources=Resources(),
    )

    [(pinned, script, _args)] = [call for name, call in backend.calls if name == "submit"]
    body = (workdir / str(script)).read_text(encoding="utf-8")
    assert pinned != "/repo"
    # The vendored root rides with the workspace's own: a house package that lives outside the
    # root is compiled inside it, so the tree a job is pinned to carries it like any other.
    assert (
        f"export PYTHONPATH={pinned}/src:{pinned}/packages/lab-core/src:"
        f"{pinned}/.mainboard/vendor/paleta-tsukuba/src" in body
    )
    assert "export PYTHONPATH=/repo/src" not in body
    assert "unset PYTHONPATH" not in body


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
    assert "provide" not in (workdir / str(script)).read_text(encoding="utf-8")


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
    # In a subshell, so the directory it changes into never becomes the job's own.
    assert (
        "( cd /repo && mainboard provide default --source /repo/.mainboard/dispatch/sources/"
        in asked
    )
    assert asked.endswith(" )")
    assert [told for told in announced if told.startswith("built default on gold for /repo")]


def test_a_host_that_will_not_build_costs_its_wave_the_head_start_and_not_the_dispatch(
    dispatcher: Dispatcher, backend: RecordingScheduler, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The job builds its own environment if it has to, so this is only ever a head start."""
    warned: list[tuple[str, tuple[object, ...]]] = []
    machine = machine_with(rules=[("provide", 1, "no pixi here")])
    monkeypatch.setattr(dispatch_module, "connection", lambda host: machine)
    monkeypatch.setattr(
        dispatch_module.logger, "warning", lambda message, *args: warned.append((message, args))
    )

    handle = dispatcher.run(
        plan(),
        shipped(dispatcher, "python -m foo"),
        root="/repo",
        resources=Resources(),
        prefix="/repo/.mainboard/prefixes/default/abcd1234",
    )

    assert handle.id == backend.submit_handle
    assert [message for message, _ in warned] == ["could not prime %s on %s: %s"]


def test_a_moving_working_tree_cannot_split_the_script_from_the_snapshot_it_runs_in(
    dispatcher: Dispatcher,
    backend: RecordingScheduler,
    workdir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One reading of the tree decides both the rendered script and the tree it is pinned in.

    A dirty tree names no commit, so its key digests the working-tree delta, and reading that
    twice across a dispatch answers twice differently the moment anything moves: the script would
    then activate from a snapshot the pin never created. A landing, which takes tens of minutes,
    is where that first cost a whole rental (vast 49867368, 2026-09-04), and a batch dispatched
    while a colleague commits is the same window on a queue.
    """
    monkeypatch.setattr(dispatch_module, "connection", lambda host: machine_with())
    deltas = iter("abcdefgh")

    def moving(*args: str) -> str:
        if "--show-toplevel" in args:
            return "/repo"
        if "describe" in args:
            return "abc1234"
        return next(deltas, "z")

    monkeypatch.setattr(provenance_module, "git", moving)
    dispatcher.run(
        plan(), shipped(dispatcher, "python -m foo"), root="/repo", resources=Resources()
    )
    [(root, script, _)] = [call for name, call in backend.calls if name == "submit"]
    assert "/sources/abc1234-dirty-" in str(root)
    assert str(root) in (workdir / str(script)).read_text(encoding="utf-8")


def test_two_dispatches_of_one_tree_share_a_snapshot_and_a_different_tree_gets_its_own(
    dispatcher: Dispatcher, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Thirty five jobs off one commit must not pay for thirty five copies of the workspace."""
    described = {"value": "v0.4.8"}
    monkeypatch.setattr(
        provenance_module,
        "git",
        lambda *args: answering(describe=described["value"], ls_files="100644 a 0\ta.py")(*args),
    )
    first = dispatcher.pinned("/repo", source=dispatcher.source())
    assert first == "/repo/.mainboard/dispatch/sources/v0.4.8"
    assert dispatcher.pinned("/repo", source=dispatcher.source()) == first
    described["value"] = "v0.4.9"
    assert dispatcher.pinned("/repo", source=dispatcher.source()) != first


def test_the_sweep_drops_the_snapshots_no_job_still_owed_an_outcome_runs_from(
    dispatcher: Dispatcher, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The other half of pinning: a host under an inode quota cannot keep every tree forever."""
    machine = machine_with("live\nnewest\nolder\nancient\n")
    monkeypatch.setattr(dispatch_module, "connection", lambda host: machine)
    dispatcher.cache.save_host(
        HostSetup(host="gold", root="/repo", synced_at="2026-09-04T00:00:00Z")
    )
    dispatcher.cache.record(run_record("H1", target="gold").model_copy(update={"source": "live"}))
    assert dispatcher.prune_sources() == {"gold": ["ancient"]}
    assert not any("live" in line for line in machine.lines if line.startswith("rm -rf"))


def test_a_host_that_was_never_mirrored_is_not_connected_to_for_a_prune(
    dispatcher: Dispatcher, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A sweep runs on a cron over every onboarded host, so the quiet ones must stay quiet."""
    reached: list[str] = []

    def connect(host: str) -> RecordingMachine:
        reached.append(host)
        return machine_with()

    monkeypatch.setattr(dispatch_module, "connection", connect)
    dispatcher.cache.save_host(HostSetup(host="gold", root="/repo"))
    dispatcher.cache.save_host(HostSetup(host="macmini", root=""))
    assert dispatcher.prune_sources() == {}
    assert reached == ["gold"]


def test_git_reports_a_local_commands_stripped_stdout(monkeypatch: pytest.MonkeyPatch) -> None:
    """And reads no stdin, since a dispatch is routinely called from inside a shell read loop."""
    asked: list[dict[str, object]] = []

    def record(*args: object, **kwargs: object) -> object:
        asked.append(kwargs)
        return type("R", (), {"stdout": " abc \n"})()

    monkeypatch.setattr(shared.subprocess, "run", record)
    assert dispatch_module.git("rev-parse", "HEAD") == "abc"
    assert [call["stdin"] for call in asked] == [shared.subprocess.DEVNULL]


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
    ("fetch_path", "source", "dest"),
    [("out/", "gold:/repo/out", ""), ("a/b/c.json", "gold:/repo/a/b/c.json", "a/b")],
)
def test_fetch_pulls_the_recorded_path_back_into_its_own_parent_directory(
    dispatcher: Dispatcher,
    workdir: Path,
    monkeypatch: pytest.MonkeyPatch,
    fetch_path: str,
    source: str,
    dest: str,
) -> None:
    """The results land under the workspace, wherever the command that pulls them was typed."""
    pulled: list[tuple[str | Sequence[str], str]] = []
    monkeypatch.setattr(
        dispatch_module, "rsync", lambda sources, target, *a, **k: pulled.append((sources, target))
    )
    dispatcher.fetch(Handle(id="H1", host="gold", root="/repo", kind="ssh", fetch_path=fetch_path))
    landing = workdir / dest
    assert pulled == [([source], f"{landing}/")]
    assert landing.is_dir()
    with pytest.raises(LookupError, match="no fetch path"):
        dispatcher.fetch(Handle(id="H1", host="gold", root="/repo", kind="ssh"))


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
    source = workdir / "outside/paleta"
    (source / "src/paleta").mkdir(parents=True)
    (source / "src/paleta/__init__.py").write_text("SHADE = 'ai'\n")
    (source / "pyproject.toml").write_text('[project]\nname = "paleta-tsukuba"\n')
    vendored = workdir / ".mainboard/vendor/paleta-tsukuba"
    vendored.mkdir(parents=True)
    for entry in sorted(source.iterdir()):
        (vendored / entry.name).symlink_to(entry)
    (source / "src/paleta/__pycache__").mkdir()
    (source / "src/paleta/__pycache__/stale.pyc").write_text("noise")
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

    arrived = landed / ".mainboard/vendor/paleta-tsukuba"
    assert arrived.is_dir() and not arrived.is_symlink()
    assert not (arrived / "src").is_symlink()
    assert (arrived / "src/paleta/__init__.py").read_text() == "SHADE = 'ai'\n"
    assert (arrived / "pyproject.toml").is_file()
    assert not (arrived / "src/paleta/__pycache__").exists()


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
