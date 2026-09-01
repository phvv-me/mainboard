import inspect
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from plumbum.commands.processes import ProcessExecutionError

from mainboard import MissionError
from mainboard.dispatch import Dispatcher, GitignoreFilter, Handle, Verdict, shared
from mainboard.dispatch import dispatcher as dispatch_module
from mainboard.dispatch.jobs import JobSpec
from mainboard.dispatch.schedulers import HostUnreachable, registry
from mainboard.dispatch.vocabulary import POLL_SECONDS, JobState, Resources
from mainboard.manifest import Container, Defaults, HostProfile, QueuePolicy

from .support import RecordingScheduler, cache, machine_with, plan, run_record

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
    monkeypatch.setattr(dispatch_module, "sleep", lambda seconds: None)
    return scheduler


@pytest.fixture
def dispatcher(workdir: Path, backend: RecordingScheduler) -> Dispatcher:
    """A dispatcher whose mirror only records what it was asked to ship, on `instance.shipped`."""
    del backend
    instance = Dispatcher(cache=cache(), sync=GitignoreFilter(workdir))
    instance.shipped: list[tuple[str, ...]] = []
    instance.rsync_up = lambda execution, root, **kwargs: instance.shipped.append(
        tuple(kwargs.get("extra", ()))
    )
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
        plan(), "python -m foo --shard 3", root="/repo", resources=resources, fetch="out/"
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
    dispatcher.run(plan(env="serving"), "python -m foo", root="/repo", resources=Resources())
    [generated] = (workdir / ".mainboard" / "dispatch" / "jobs").glob("job-*.sh")
    text = generated.read_text()
    assert "/repo/.mainboard/activate-serving.sh" in text
    assert "/repo/.mainboard/envs/serving/.pixi/envs/serving/bin" in text


def test_run_on_a_pbs_host_with_no_resolved_walltime_fails_before_any_sync(
    dispatcher: Dispatcher, backend: RecordingScheduler
) -> None:
    """No declared default and no resolved walltime is a clear error, never a site constant."""
    pbs = plan(profile=HostProfile(kind="pbs", root="/repo", sync={"include": ["src"]}))
    with pytest.raises(ValueError, match="explicit walltime"):
        dispatcher.run(pbs, "python -m foo", root="/repo", resources=Resources())
    assert dispatcher.shipped == []
    assert backend.calls == []


def test_run_containerized_wraps_the_command_via_the_builder_or_refuses_without_one(
    dispatcher: Dispatcher, backend: RecordingScheduler, workdir: Path
) -> None:
    containerized = plan(**_CONTAINERIZED)
    with pytest.raises(LookupError, match="no container argv builder"):
        dispatcher.run(containerized, "python -m foo", root="/repo", resources=Resources())
    dispatcher.run(
        containerized,
        "python -m foo",
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
        dispatch_module, "git", lambda *args: "abc1234" if args[0] == "rev-parse" else porcelain
    )
    handle = dispatcher.submit(
        plan(), "/repo", script="train.sh", args=("--x", "1"), resources=Resources()
    )
    [run] = dispatcher.cache.recent(10)
    assert (run.handle, run.target, run.git_sha, run.dirty) == (handle, "gold", "abc1234", dirty)
    assert run.args == "--x 1"


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


def _repo(path: Path, dirty: bool) -> None:
    """A real git repository at `path` with one commit, optionally carrying uncommitted work."""
    path.mkdir(parents=True, exist_ok=True)
    run = lambda *args: subprocess.run(  # noqa: E731
        ["git", "-C", str(path), *args], check=True, capture_output=True, text=True
    )
    run("init", "-q")
    run("config", "user.email", "t@t")
    run("config", "user.name", "t")
    (path / "tracked.py").write_text("x = 1\n")
    run("add", "tracked.py")
    run("commit", "-q", "-m", "one")
    if dirty:
        (path / "tracked.py").write_text("x = 2\n")


def test_the_source_stamp_names_the_repository_that_owns_the_job_not_the_submitters_cwd(
    tmp_path: Path,
) -> None:
    _repo(tmp_path, dirty=True)
    _repo(tmp_path / "inner", dirty=False)
    nested = dispatch_module.source_of("pytest inner/tracked.py -q", tmp_path)
    fallback = dispatch_module.source_of("python -m foo", tmp_path)
    assert nested and "-dirty" not in nested
    assert fallback.endswith("-dirty")


def test_a_token_naming_a_path_outside_any_repository_is_skipped_not_a_dead_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The first token can name a real path that git owns nothing of; the scan tries the next."""
    (tmp_path / "plain").mkdir()
    (tmp_path / "plain" / "data.txt").write_text("not tracked by anything")
    repository = tmp_path / "repo"
    repository.mkdir()
    (repository / "tracked.py").write_text("x = 1\n")

    def git(*args: str) -> str:
        if "rev-parse" in args:
            return "" if args[1] == str(tmp_path / "plain") else str(repository)
        return "clean-source"

    monkeypatch.setattr(dispatch_module, "git", git)
    found = dispatch_module.source_of("cmd plain/data.txt repo/tracked.py", tmp_path)
    assert found == "clean-source"
