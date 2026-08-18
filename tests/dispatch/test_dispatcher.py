import inspect
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from plumbum.commands.processes import ProcessExecutionError

from mainboard import ExecutionPlan, MissionError
from mainboard.dispatch import Dispatcher, Handle, Verdict
from mainboard.dispatch import dispatcher as dispatch_module
from mainboard.dispatch import schedulers as schedulers_module
from mainboard.dispatch.jobs import JobSpec
from mainboard.dispatch.schedulers import HostUnreachable, JobState, Resources
from mainboard.dispatch.schedulers.base import POLL_SECONDS
from mainboard.dispatch.state import RunRecord
from mainboard.manifest import Container, Defaults, HostProfile, QueuePolicy

from .conftest import FakeRemote, RecordingScheduler

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mainboard.dispatch.transport import Machine

type PlanField = str | HostProfile | Container | dict[str, str] | None


def plan(**overrides: PlanField) -> ExecutionPlan:
    fields: dict[str, PlanField] = {
        "host": "gold",
        "profile": HostProfile(kind="ssh", root="/repo", sync={"include": ["src"]}),
        "env": "default",
    }
    fields.update(overrides)
    return ExecutionPlan.model_validate(fields)


class _StubStrategy:
    """A `Strategy`-shaped double that always resolves to one canned scheduler."""

    def __init__(self, scheduler: RecordingScheduler) -> None:
        self.scheduler = scheduler

    def select(self, kind: str, default: str | None = None) -> RecordingScheduler:
        del kind, default
        return self.scheduler


@pytest.fixture
def backend(monkeypatch: pytest.MonkeyPatch) -> RecordingScheduler:
    """Pin the dispatch seams: pick/SCHEDULERS -> a recording backend, connection -> a fake remote."""  # ruff:ignore[line-too-long]  reason=descriptive fixture docstring since=2026-08-16
    sched = RecordingScheduler()
    monkeypatch.setattr(dispatch_module, "pick", lambda profile: sched)
    monkeypatch.setattr(schedulers_module, "SCHEDULERS", _StubStrategy(sched))
    monkeypatch.setattr(dispatch_module, "connection", lambda host: FakeRemote())
    monkeypatch.setattr(
        dispatch_module, "git", lambda *a: "abc1234" if a[0] == "rev-parse" else ""
    )
    return sched


@pytest.fixture
def dispatcher(workdir: Path, backend: RecordingScheduler) -> Dispatcher:
    del backend
    instance = Dispatcher(cache=dispatch_module.Cache(workdir / "db.sqlite"))
    instance.rsync_up = lambda *a, **k: None  # type: ignore[method-assign]  reason=test double stands in for the bound method since=2026-08-16
    return instance


# --- Handle / Verdict ---


def test_verdict_projects_to_ok_and_exit_code() -> None:
    assert Verdict(verdict="ok", exit_code=0).ok is True
    assert Verdict(verdict="ok").code == 0
    assert Verdict(verdict="failed", exit_code=1).ok is False
    assert Verdict(verdict="failed").code == 1
    assert Verdict(verdict="vanished").code == 3


def test_handle_carries_host_root_and_kind() -> None:
    handle = Handle(id="H1", host="gold", root="/repo", kind="ssh", fetch_path="out/")
    assert handle.fetch_path == "out/"


# --- run ---


def test_run_dispatches_and_returns_a_handle(
    dispatcher: Dispatcher, backend: RecordingScheduler
) -> None:
    handle = dispatcher.run(
        plan(), "python -m foo --shard 3", root="/repo", resources=Resources(gpus=1), fetch="out/"
    )
    assert isinstance(handle, Handle)
    assert handle.id == "H1"
    assert handle.host == "gold"
    assert handle.fetch_path == "out/"
    [(_root, script, _args)] = [v for k, v in backend.calls if k == "submit"]
    assert script.startswith(".mainboard/dispatch/jobs/")


def test_run_renders_the_job_script_against_the_plans_own_environment(
    dispatcher: Dispatcher, workdir: Path
) -> None:
    """A job queued for `serving` must activate serving, not whatever was installed last."""
    dispatcher.run(plan(env="serving"), "python -m foo", root="/repo", resources=Resources())
    [generated] = (workdir / ".mainboard" / "dispatch" / "jobs").glob("job-*.sh")
    text = generated.read_text()
    assert "/repo/.mainboard/activate-serving.sh" in text
    assert "/repo/.mainboard/.pixi/envs/serving/bin" in text


def test_run_ships_the_generated_script_as_an_extra_path(
    workdir: Path, backend: RecordingScheduler
) -> None:
    shipped: list[tuple[str, ...]] = []
    instance = Dispatcher()
    instance.rsync_up = lambda p, root, **k: shipped.append(tuple(k.get("extra", ())))  # type: ignore[method-assign]  reason=test double stands in for the bound method since=2026-08-16
    handle = instance.run(plan(), "python -m foo", root="/repo", resources=Resources())
    [(generated,)] = shipped
    assert generated.startswith(".mainboard/dispatch/jobs/")
    [(_root, script, _args)] = [v for k, v in backend.calls if k == "submit"]
    assert script == generated
    assert handle.id == "H1"


def test_run_threads_resources_to_the_backend(
    dispatcher: Dispatcher, backend: RecordingScheduler
) -> None:
    resources = Resources(gpus=4, walltime="01:00:00", queue="gen-S", mem_gb=240)
    dispatcher.run(plan(), "python -m foo", root="/repo", resources=resources)
    sent = backend.submit_resources
    assert sent.gpus == 4
    assert sent.walltime == "01:00:00"
    assert sent.queue == "gen-S"
    assert sent.mem_gb == 240


def test_run_on_a_pbs_host_with_no_resolved_walltime_fails_before_any_sync(
    workdir: Path, backend: RecordingScheduler
) -> None:
    """No manifest-declared default and no caller-resolved walltime is a clear error, never a
    silently-injected site constant."""
    instance = Dispatcher()
    called: list[tuple[ExecutionPlan | str, ...]] = []
    instance.rsync_up = lambda *a, **k: called.append(a)  # type: ignore[method-assign]  reason=test double stands in for the bound method since=2026-08-16
    pbs_plan = plan(profile=HostProfile(kind="pbs", root="/repo", sync={"include": ["src"]}))
    with pytest.raises(ValueError, match="explicit walltime"):
        instance.run(pbs_plan, "python -m foo", root="/repo", resources=Resources())
    assert called == []
    assert backend.calls == []


def test_run_containerized_without_a_builder_raises_before_rendering(workdir: Path) -> None:
    instance = Dispatcher()
    host = HostProfile(kind="ssh", root="/repo", container="ngc", sync={"include": ["src"]})
    container_plan = plan(
        profile=host, container=Container(image="nvcr.io/nvidia/pytorch:25.06-py3")
    )
    with pytest.raises(LookupError, match="no container argv builder"):
        instance.run(container_plan, "python -m foo", root="/repo", resources=Resources())


def test_run_containerized_wraps_the_command_via_the_injected_builder(
    dispatcher: Dispatcher, backend: RecordingScheduler, workdir: Path
) -> None:
    host = HostProfile(kind="ssh", root="/repo", container="ngc", sync={"include": ["src"]})
    container_plan = plan(
        profile=host, container=Container(image="nvcr.io/nvidia/pytorch:25.06-py3")
    )
    dispatcher.run(
        container_plan,
        "python -m foo",
        root="/repo",
        resources=Resources(),
        containerize=lambda inner: ["apptainer", "exec", "image.sif", *inner],
    )
    [(_root, script, _args)] = [v for k, v in backend.calls if k == "submit"]
    text = (workdir / script).read_text()
    assert text.splitlines()[-1] == "apptainer exec image.sif bash -c 'python -m foo'"


# --- submit ---


def test_submit_calls_admission_before_any_ssh(workdir: Path, backend: RecordingScheduler) -> None:
    host = HostProfile(
        kind="ssh",
        root="/repo",
        sync={"include": ["src"]},
        queues={"short-g": QueuePolicy(max_walltime="00:10:00")},
    )
    instance = Dispatcher()
    touched: list[str] = []
    instance.rsync_up = lambda *a, **k: touched.append("rsync")  # type: ignore[method-assign]  reason=test double stands in for the bound method since=2026-08-16
    with pytest.raises(MissionError, match="exceeds the 'short-g' ceiling"):
        instance.submit(
            plan(profile=host),
            "/repo",
            script="job.sh",
            args=(),
            resources=Resources(queue="short-g", walltime="08:00:00"),
        )
    assert touched == []
    assert backend.calls == []


def test_submit_records_the_run_with_git_provenance(
    dispatcher: Dispatcher, backend: RecordingScheduler
) -> None:
    handle = dispatcher.submit(plan(), "/repo", script="train.sh", args=(), resources=Resources())
    run = dispatcher.cache.recent(10)[0]
    assert run.handle == handle
    assert run.target == "gold"
    assert run.git_sha == "abc1234"
    assert run.dirty == 0


def test_submit_dirty_tree_is_recorded(
    dispatcher: Dispatcher, backend: RecordingScheduler, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        dispatch_module, "git", lambda *a: "abc1234" if a[0] == "rev-parse" else "M x.py"
    )
    dispatcher.submit(plan(), "/repo", script="train.sh", args=(), resources=Resources())
    [run] = dispatcher.cache.recent(10)
    assert run.dirty == 1


def test_submit_verify_failure_aborts_before_the_scheduler(
    dispatcher: Dispatcher, backend: RecordingScheduler, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        dispatch_module,
        "connection",
        lambda host: FakeRemote(healthy=False, stderr="ModuleNotFoundError"),
    )
    with pytest.raises(SystemExit, match="environment on 'gold' is broken"):
        dispatcher.submit(plan(), "/repo", script="train.sh", args=(), resources=Resources())
    assert backend.calls == []


def test_submit_scheduler_rejection_names_the_host(
    dispatcher: Dispatcher, backend: RecordingScheduler
) -> None:
    def reject(
        remote: Machine, root: str, *, script: str, args: Sequence[str], resources: Resources
    ) -> str:
        raise SystemExit("no PBS queue resolved")

    backend.submit = reject  # type: ignore[method-assign]  reason=test double stands in for the bound method since=2026-08-16
    with pytest.raises(SystemExit, match="submission to host 'gold' failed"):
        dispatcher.submit(plan(), "/repo", script="train.sh", args=(), resources=Resources())


# --- await_many ---


def test_await_many_blocks_until_each_handle_is_terminal(
    dispatcher: Dispatcher, backend: RecordingScheduler, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(dispatch_module, "sleep", lambda s: None)
    backend.state_result = JobState(handle="H1", state="F", exit_code=0, verdict="ok")
    handle = Handle(id="H1", host="gold", root="/repo", kind="ssh")
    verdicts = dispatcher.await_many([handle])
    assert verdicts[handle].ok
    assert verdicts[handle].exit_code == 0


def test_await_many_polls_running_handles_until_they_finish(
    dispatcher: Dispatcher, backend: RecordingScheduler, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(dispatch_module, "sleep", lambda s: None)
    running = JobState(handle="H1", state="R", verdict="running")
    done = JobState(handle="H1", state="F", exit_code=0, verdict="ok")
    states = iter([running, running, done])
    backend.state = lambda remote, root, handle: next(states)  # type: ignore[method-assign]  reason=test double stands in for the bound method since=2026-08-16
    handle = Handle(id="H1", host="gold", root="/repo", kind="ssh")
    verdicts = dispatcher.await_many([handle])
    assert verdicts[handle].verdict == "ok"


def test_await_many_retries_a_transient_blip(
    dispatcher: Dispatcher, backend: RecordingScheduler, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(dispatch_module, "sleep", lambda s: None)
    calls = {"n": 0}

    def state(remote: Machine, root: str, *, handle: str) -> JobState:
        calls["n"] += 1
        if calls["n"] == 1:
            raise HostUnreachable("blip")
        return JobState(handle="H1", state="F", exit_code=0, verdict="ok")

    backend.state = state  # type: ignore[method-assign]  reason=test double stands in for the bound method since=2026-08-16
    handle = Handle(id="H1", host="gold", root="/repo", kind="ssh")
    verdicts = dispatcher.await_many([handle])
    assert verdicts[handle].ok
    assert calls["n"] == 2


def test_await_many_carries_the_failure_reason(
    dispatcher: Dispatcher, backend: RecordingScheduler, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(dispatch_module, "sleep", lambda s: None)
    backend.state_result = JobState(handle="H1", state="F", exit_code=137, verdict="failed")
    monkeypatch.setattr(dispatch_module, "read_log", lambda remote, root, handle: "warming up\n")
    handle = Handle(id="H1", host="gold", root="/repo", kind="ssh")
    verdict = dispatcher.await_many([handle])[handle]
    assert verdict.verdict == "failed"
    assert "memory" in verdict.reason


def test_await_many_persists_a_terminal_verdict_to_the_cache(
    dispatcher: Dispatcher, backend: RecordingScheduler, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(dispatch_module, "sleep", lambda s: None)
    dispatcher.cache.record(
        RunRecord(
            handle="H1",
            target="gold",
            kind="ssh",
            script="a.sh",
            args="",
            git_sha="x",
            dirty=0,
            submitted_at="t0",
        )
    )
    backend.state_result = JobState(handle="H1", state="F", exit_code=0, verdict="ok")
    dispatcher.await_many([Handle(id="H1", host="gold", root="/repo", kind="ssh")])
    assert dispatcher.cache.run("H1").verdict == "ok"


def test_await_many_tolerates_an_unrecorded_handle(
    dispatcher: Dispatcher, backend: RecordingScheduler, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(dispatch_module, "sleep", lambda s: None)
    backend.state_result = JobState(handle="H1", state="F", exit_code=0, verdict="ok")
    handle = Handle(id="H1", host="gold", root="/repo", kind="ssh")
    assert dispatcher.await_many([handle])[handle].ok


def test_await_many_default_interval_is_poll_seconds() -> None:
    assert inspect.signature(Dispatcher.await_many).parameters["interval"].default == POLL_SECONDS


# --- probe / state ---


def test_probe_absorbs_an_unreachable_host_while_state_names_it(
    dispatcher: Dispatcher, backend: RecordingScheduler
) -> None:
    def down(remote: Machine, root: str, *, handle: str) -> JobState:
        raise HostUnreachable("ssh connect to 'gold' failed: connection timed out")

    backend.state = down  # type: ignore[method-assign]  reason=test double stands in for the bound method since=2026-08-18
    handle = Handle(id="H1", host="gold", root="/repo", kind="ssh")
    assert dispatcher.probe(handle) is None
    with pytest.raises(HostUnreachable, match="timed out"):
        dispatcher.state(handle)


# --- fetch / fetch_path ---


def test_fetch_pulls_the_recorded_path_back(
    monkeypatch: pytest.MonkeyPatch, dispatcher: Dispatcher
) -> None:
    calls: list[tuple[str | Sequence[str], str]] = []
    monkeypatch.setattr(
        dispatch_module, "rsync", lambda sources, dest, *a, **k: calls.append((sources, dest))
    )
    dispatcher.fetch(Handle(id="H1", host="gold", root="/repo", kind="ssh", fetch_path="out/"))
    [(sources, dest)] = calls
    assert sources == ["gold:/repo/out"]
    assert dest == "./"


def test_fetch_of_a_single_file_lands_in_its_parent(
    workdir: Path, monkeypatch: pytest.MonkeyPatch, dispatcher: Dispatcher
) -> None:
    calls: list[tuple[str | Sequence[str], str]] = []
    monkeypatch.setattr(
        dispatch_module, "rsync", lambda sources, dest, *a, **k: calls.append((sources, dest))
    )
    dispatcher.fetch(
        Handle(id="H1", host="gold", root="/repo", kind="ssh", fetch_path="a/b/c.json")
    )
    assert (workdir / "a/b").is_dir()
    [(sources, dest)] = calls
    assert sources == ["gold:/repo/a/b/c.json"]
    assert dest == "a/b/"


def test_fetch_without_a_path_is_a_lookup_error(dispatcher: Dispatcher) -> None:
    with pytest.raises(LookupError, match="no fetch path"):
        dispatcher.fetch(Handle(id="H1", host="gold", root="/repo", kind="ssh"))


# --- write_job_script ---


def test_write_job_script_is_content_addressed(dispatcher: Dispatcher, workdir: Path) -> None:
    spec = JobSpec(cmd="python -m foo", env_prefix="/repo/.mainboard/envs/default")
    first = dispatcher.write_job_script(spec, pbs=False)
    assert first.startswith(".mainboard/dispatch/jobs/")
    again = dispatcher.write_job_script(spec, pbs=False)
    assert again == first


# --- _prepare_script ---


def test_prepare_script_leaves_a_bare_name_unchanged(dispatcher: Dispatcher) -> None:
    prepared, staged = dispatcher._prepare_script("job")  # ruff:ignore[private-member-access]  reason=unit-tests the module-private helper since=2026-08-16
    assert prepared == "job"
    assert staged == ()


def test_prepare_script_stages_an_explicit_existing_path(
    dispatcher: Dispatcher, workdir: Path
) -> None:
    external = workdir.parent / "external.sh"
    external.write_text("#!/bin/bash\necho hi\n")
    prepared, staged = dispatcher._prepare_script(str(external))  # ruff:ignore[private-member-access]  reason=unit-tests the module-private helper since=2026-08-16
    assert prepared.startswith(".mainboard/dispatch/jobs/")
    assert staged == (prepared,)
    assert (workdir / prepared).read_text() == external.read_text()


def test_prepare_script_rejects_a_missing_explicit_path(
    dispatcher: Dispatcher, workdir: Path
) -> None:
    with pytest.raises(FileNotFoundError, match="cannot be shipped to the host"):
        dispatcher._prepare_script("./missing/job.sh")  # ruff:ignore[private-member-access]  reason=unit-tests the module-private helper since=2026-08-16


# --- rsync_up ---


def test_rsync_up_fails_fast_on_empty_include(workdir: Path) -> None:
    instance = Dispatcher()
    host = HostProfile(kind="ssh", root="/repo", sync={"include": []})
    with pytest.raises(LookupError, match="nothing to sync"):
        instance.rsync_up(plan(profile=host), "/repo")


def test_rsync_up_drops_stale_include_paths_with_one_warning(
    workdir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (workdir / "src").mkdir()
    host = HostProfile(kind="ssh", root="/repo", sync={"include": ["src", "packages/meteng"]})
    instance = Dispatcher()
    captured: dict[str, str | Sequence[str]] = {}
    monkeypatch.setattr(
        dispatch_module,
        "rsync",
        lambda sources, dest, flags, **k: captured.update(sources=sources),
    )
    warned: list[tuple[str, tuple[int | str, ...]]] = []
    monkeypatch.setattr(dispatch_module.logger, "warning", lambda msg, *a: warned.append((msg, a)))
    instance.rsync_up(plan(profile=host), "/repo")
    assert captured["sources"][0] == "src"
    [(message, args)] = warned
    assert "stale sync include" in message
    assert args[0] == 1
    assert args[1] == "packages/meteng"


def test_rsync_up_with_every_include_missing_is_a_lookup_error(workdir: Path) -> None:
    host = HostProfile(kind="ssh", root="/repo", sync={"include": ["gone"]})
    instance = Dispatcher()
    with pytest.raises(LookupError, match="missing locally"):
        instance.rsync_up(plan(profile=host), "/repo")


def test_rsync_up_rejects_an_incomplete_required_pair(workdir: Path) -> None:
    (workdir / "src").mkdir()
    envdir = workdir / ".mainboard/envs/default"
    envdir.mkdir(parents=True)
    (envdir / "pixi.toml").write_text("x")
    host = HostProfile(kind="ssh", root="/repo", sync={"include": ["src"]})
    instance = Dispatcher()
    with pytest.raises(LookupError, match="incomplete"):
        instance.rsync_up(
            plan(profile=host),
            "/repo",
            required=[(".mainboard/envs/default/pixi.toml", ".mainboard/envs/default/pixi.lock")],
        )


def test_rsync_up_punches_through_the_denylist_for_a_required_pair(
    workdir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (workdir / "src").mkdir()
    envdir = workdir / ".mainboard/envs/default"
    envdir.mkdir(parents=True)
    (envdir / "pixi.toml").write_text("x")
    (envdir / "pixi.lock").write_text("y")
    host = HostProfile(kind="ssh", root="/repo", sync={"include": ["src"]})
    instance = Dispatcher()
    captured: dict[str, str | Sequence[str] | bool] = {}
    monkeypatch.setattr(
        dispatch_module,
        "rsync",
        lambda sources, dest, flags, **k: captured.update(sources=sources, **k),
    )
    instance.rsync_up(
        plan(profile=host),
        "/repo",
        required=[(".mainboard/envs/default/pixi.toml", ".mainboard/envs/default/pixi.lock")],
    )
    assert ".mainboard/envs/default/pixi.toml" in captured["sources"]
    assert "/.mainboard/" in captured["include"]
    assert "/.mainboard/***" in captured["exclude"]
    assert captured["allow_vanished"] is False


def test_rsync_up_preserves_an_ordinary_mirror_error(
    workdir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (workdir / "src").mkdir()
    host = HostProfile(kind="ssh", root="/repo", sync={"include": ["src"]})
    instance = Dispatcher()
    failure = ProcessExecutionError(["rsync"], 23, "", "partial transfer")

    def fail(*a, **k) -> None:
        raise failure

    monkeypatch.setattr(dispatch_module, "rsync", fail)
    with pytest.raises(ProcessExecutionError) as caught:
        instance.rsync_up(plan(profile=host), "/repo")
    assert caught.value is failure


def test_rsync_up_wraps_a_failed_required_transfer(
    workdir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (workdir / "src").mkdir()
    host = HostProfile(kind="ssh", root="/repo", sync={"include": ["src"]})
    instance = Dispatcher()

    def fail(*a, **k) -> None:
        raise ProcessExecutionError(["rsync"], 12, "", "connection reset")

    monkeypatch.setattr(dispatch_module, "rsync", fail)
    with pytest.raises(RuntimeError, match="submission aborted before scheduler dispatch"):
        instance.rsync_up(
            plan(profile=host), "/repo", extra=(".mainboard/dispatch/jobs/job-x.sh",)
        )


# --- git ---


def test_git_strips_stdout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        dispatch_module.subprocess, "run", lambda *a, **k: type("R", (), {"stdout": " abc \n"})()
    )
    assert dispatch_module.git("rev-parse", "HEAD") == "abc"


# --- de-hardcoded defaults ---


def test_manifest_owns_the_walltime_default_not_dispatch_code() -> None:
    """The `00:30:00` fallback lives on the manifest schema's `Defaults`, never in dispatch."""
    assert Defaults().walltime == "00:30:00"
    source = Path(dispatch_module.__file__).read_text(encoding="utf-8")
    assert "debug-g" not in source
    assert "00:30:00" not in source


def test_handles_accept_a_numeric_scheduler_id_and_store_text() -> None:
    assert Handle(id=42, host="gold", root="/repo", kind="ssh").id == "42"
    assert JobState(handle=42, verdict="running").handle == "42"
