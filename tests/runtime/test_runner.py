import json
import os
import runpy
import shlex
import signal
import sys
from collections.abc import Mapping
from pathlib import Path

import pytest

from mainboard.cli import build
from mainboard.dispatch.evidence import RECEIPTS_VAR, receipts_in
from mainboard.runtime.entry import Entering, Refusal
from mainboard.runtime.job import Job, ToolCall, WorkspaceActivation
from mainboard.runtime.runner import Receipts, Runner

posix_only = pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals")

# One trial receipt, long enough that a provider's line cut would have torn it.
_RECEIPT = json.dumps({"trial_receipt": {"run_id": "completed", "pad": "x" * 900}})


def python(code: str) -> str:
    """A command line running `code` in this interpreter, on any platform's runner."""
    return shlex.join([sys.executable, "-c", code])


def writing(status: int = 0) -> str:
    """A command writing one receipt to the file it was pointed at, then exiting `status`."""
    return python(
        "import os, sys\n"
        f"open(os.environ[{RECEIPTS_VAR!r}], 'a').write({_RECEIPT!r} + '\\n')\n"
        "print('the command ran')\n"
        f"sys.exit({status})"
    )


class Entered(Entering):
    """A machine where entering an environment hands back the base, or refuses."""

    def __init__(self, refusal: Refusal | None = None) -> None:
        self.refusal = refusal

    def ready(self, shard: Path, script: Path) -> bool:
        return True

    def activated(
        self, base: Mapping[str, str], *, shard: Path, script: Path, env: str, cwd: str
    ) -> dict[str, str]:
        if self.refusal is not None:
            raise self.refusal
        return {**base, "ENTERED": env}

    def executables(self, prefix: Path) -> list[Path]:
        return []


def job(
    tmp_path: Path,
    command: str,
    **fields: str | bool | dict[str, str] | ToolCall | tuple[str, ...],
) -> Job:
    """A job running `command` from `tmp_path` in the default environment of its workspace."""
    installed = tmp_path / ".mainboard" / "envs" / "default" / ".pixi" / "envs" / "default"
    return Job.model_validate(
        {
            "command": command,
            "root": str(tmp_path),
            "activation": WorkspaceActivation(
                script=str(tmp_path / ".mainboard" / "activate.sh"),
                prefix=str(installed),
                refusal="install default",
            ),
            **fields,
        }
    )


def runner(
    tmp_path: Path,
    command: str,
    *,
    how: Entering | None = None,
    **fields: str | bool | dict[str, str] | ToolCall | tuple[str, ...],
) -> Runner:
    """A runner for `job(tmp_path, command, **fields)` on a machine entering nothing at all."""
    return Runner(job(tmp_path, command, **fields), how=how or Entered(), grace=0.5)


@pytest.mark.parametrize("status", [0, 7], ids=["success", "failure"])
def test_a_job_frames_its_receipts_back_once_and_keeps_the_commands_status(
    tmp_path: Path, capfd: pytest.CaptureFixture[str], status: int
) -> None:
    """Framing the receipts back costs the job none of its exit code."""
    assert runner(tmp_path, writing(status)).run() == status
    output = capfd.readouterr().out
    assert "the command ran" in output
    assert receipts_in(output) == (_RECEIPT,)
    assert output.count("mainboard-receipts-begin") == 1
    assert not list(Path(Receipts().path.parent).glob(f"mainboard-receipts.{os.getpid()}.*"))


def test_a_job_that_wrote_no_receipts_leaves_its_output_as_it_was(
    tmp_path: Path, capfd: pytest.CaptureFixture[str]
) -> None:
    assert runner(tmp_path, python("pass")).run() == 0
    assert "mainboard-receipts" not in capfd.readouterr().out
    removed = python(f"import os; os.remove(os.environ[{RECEIPTS_VAR!r}])")
    assert runner(tmp_path, removed).run() == 0
    assert "mainboard-receipts" not in capfd.readouterr().out


def test_a_refused_environment_ends_the_job_before_the_command_with_its_own_status(
    tmp_path: Path, capfd: pytest.CaptureFixture[str]
) -> None:
    """A failed activation leaves no invented receipt, and says why on stderr."""
    marker = tmp_path / "ran"
    command = python(f"open({str(marker)!r}, 'w')")
    assert runner(tmp_path, command, how=Entered(Refusal("install default"))).run() == 1
    assert runner(tmp_path, command, how=Entered(Refusal("", 23))).run() == 23
    captured = capfd.readouterr()
    assert captured.err == "install default\n"
    assert "mainboard-receipts" not in captured.out
    assert not marker.exists()


def test_a_pbs_job_appends_its_output_and_status_where_a_later_poll_reads_them(
    tmp_path: Path, capfd: pytest.CaptureFixture[str]
) -> None:
    """A server that purges the job from its history still leaves its exit behind."""
    logs = tmp_path / ".mainboard" / "dispatch" / "logs"
    ran = Runner(
        job(tmp_path, writing(5), logs=str(logs)),
        environ={**os.environ, "PBS_JOBID": "42.opbs"},
        how=Entered(),
    )
    assert ran.run() == 5
    log = (logs / "42.log").read_text(encoding="utf-8")
    assert "the command ran" in log
    assert receipts_in(log) == (_RECEIPT,)
    assert log.endswith("exit=5\n")
    assert (logs / "42.exit").read_text(encoding="utf-8") == "exit=5\n"
    assert capfd.readouterr().out == ""
    refused = Runner(
        job(tmp_path, python("pass"), logs=str(logs)),
        environ={"PBS_JOBID": "43"},
        how=Entered(Refusal("install default")),
    )
    assert refused.run() == 1
    assert (logs / "43.log").read_text(encoding="utf-8") == "install default\nexit=1\n"


def test_the_walltime_ends_the_command_and_the_log_says_so(
    tmp_path: Path, capfd: pytest.CaptureFixture[str]
) -> None:
    """A triage view decodes the stop instead of showing a raw SIGTERM backtrace."""
    sleeping = python(
        f"import os; open(os.environ[{RECEIPTS_VAR!r}], 'w').write({_RECEIPT!r}); "
        "import time; time.sleep(60)"
    )
    assert runner(tmp_path, sleeping, walltime="00:00:01").run() == 124
    output = capfd.readouterr().out
    assert "mainboard: killed at walltime 00:00:01 (exit 124)" in output
    assert receipts_in(output) == (_RECEIPT,)


@posix_only
def test_a_command_that_ignores_the_walltime_is_killed_after_its_grace(
    tmp_path: Path, capfd: pytest.CaptureFixture[str]
) -> None:
    stubborn = python(
        "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)"
    )
    assert runner(tmp_path, stubborn, walltime="00:00:01").run() == 137
    assert "killed at walltime 00:00:01 (exit 137)" in capfd.readouterr().out


@posix_only
def test_a_terminated_job_ends_its_command_frames_its_receipts_and_exits_143(
    tmp_path: Path, capfd: pytest.CaptureFixture[str]
) -> None:
    """What `trap 'exit 143' TERM` gave a job, whatever the command itself exited with."""
    previous = signal.getsignal(signal.SIGTERM)
    ending = python(
        f"import os, signal, time; open(os.environ[{RECEIPTS_VAR!r}], 'w').write({_RECEIPT!r}); "
        "os.kill(os.getppid(), signal.SIGTERM); time.sleep(60)"
    )
    assert runner(tmp_path, ending).run() == 143
    assert receipts_in(capfd.readouterr().out) == (_RECEIPT,)
    assert signal.getsignal(signal.SIGTERM) == previous


def test_a_job_told_to_stop_before_its_command_never_starts_it(tmp_path: Path) -> None:
    marker = tmp_path / "ran"
    stopped = runner(tmp_path, python(f"open({str(marker)!r}, 'w')"))
    stopped.ended(signal.SIGTERM, None)
    assert stopped.run() == 128 + signal.SIGTERM
    assert not marker.exists()


def test_the_command_sees_its_exports_after_the_environment_and_its_own_pythonpath(
    tmp_path: Path,
) -> None:
    seen = tmp_path / "seen.json"
    dump = python(f"import json, os; json.dump(dict(os.environ), open({str(seen)!r}, 'w'))")
    exports = {"MAINBOARD_SOURCE": "abc", "ENTERED": "overridden by the host"}
    environ = {**os.environ, "PYTHONPATH": "/inherited", RECEIPTS_VAR: "/stale"}
    Runner(job(tmp_path, dump, variables=exports), environ=environ, how=Entered()).run()
    isolated = json.loads(seen.read_text(encoding="utf-8"))
    assert (
        isolated["ENTERED"] == "overridden by the host" and isolated["MAINBOARD_SOURCE"] == "abc"
    )
    assert "PYTHONPATH" not in isolated
    assert isolated[RECEIPTS_VAR] != "/stale"
    Runner(job(tmp_path, dump, pythonpath="/pinned/src"), environ=environ, how=Entered()).run()
    assert json.loads(seen.read_text(encoding="utf-8"))["PYTHONPATH"] == "/pinned/src"
    Runner(job(tmp_path, dump, isolate_pythonpath=False), environ=environ, how=Entered()).run()
    assert json.loads(seen.read_text(encoding="utf-8"))["PYTHONPATH"] == "/inherited"


def test_the_calls_around_the_command_run_this_tool_with_their_own_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    """Provisioning, attestation and the sampler each reach the tool running the job."""
    record = (
        "import os, sys; "
        "open(sys.argv[1], 'w').write(os.environ.get('TOKEN', '') + ' ' + os.getcwd())"
    )
    monkeypatch.setattr(Runner, "tool", (sys.executable, "-c", record))
    credentials = tmp_path / "tracking.json"
    credentials.write_text(json.dumps({"TOKEN": "secret"}), encoding="utf-8")
    mirror = tmp_path / "mirror"
    mirror.mkdir()
    ran = runner(
        tmp_path,
        python("pass"),
        provide=ToolCall(args=(str(tmp_path / "provided"),), cwd=str(mirror)),
        attestation=ToolCall(args=(str(tmp_path / "attested"),), credentials=str(credentials)),
        sampler=ToolCall(args=(str(tmp_path / "sampled"),), credentials=str(credentials)),
    )
    assert ran.run() == 0
    assert (tmp_path / "provided").read_text(encoding="utf-8") == f" {mirror}"
    assert (tmp_path / "attested").read_text(encoding="utf-8") == f"secret {tmp_path}"
    assert "could not build" not in capfd.readouterr().out


def test_a_failed_build_is_said_and_left_to_the_environment_to_refuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(Runner, "tool", (sys.executable, "-c", "raise SystemExit(3)"))
    ran = runner(tmp_path, python("pass"), provide=ToolCall(args=("provide", "default")))
    assert ran.run() == 0
    assert (
        "mainboard: could not build the environment this job was dispatched with"
        in capfd.readouterr().out
    )


def test_credentials_that_were_never_staged_add_nothing(tmp_path: Path) -> None:
    assert Runner.credentials(ToolCall(args=())) == {}
    assert Runner.credentials(ToolCall(args=(), credentials=str(tmp_path / "missing"))) == {}


def test_the_command_runs_as_its_container_bash_or_its_own_words(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Windows has no shell to hand a line to, so the line is split and its program looked up."""
    contained = runner(
        tmp_path, "run", container=("apptainer", "exec", "i.sif", "bash", "-c", "run")
    )
    assert contained.argv({}) == ["apptainer", "exec", "i.sif", "bash", "-c", "run"]
    monkeypatch.setattr("platform.system", lambda: "Linux")
    assert runner(tmp_path, "python -m x 'a b'").argv({}) == ["bash", "-c", "python -m x 'a b'"]
    monkeypatch.setattr("platform.system", lambda: "Windows")
    found = Path(sys.executable)
    windows = runner(tmp_path, f"{found.name} -m x 'a b'")
    assert windows.argv({"PATH": str(found.parent)}) == [str(found), "-m", "x", "a b"]
    assert runner(tmp_path, "nowhere-to-be-found --x").argv({"PATH": ""}) == [
        "nowhere-to-be-found",
        "--x",
    ]


def test_the_cli_hands_a_record_to_the_runner_and_exits_with_its_status(
    tmp_path: Path,
) -> None:
    installed = tmp_path / ".mainboard" / "envs" / "default" / ".pixi" / "envs" / "default"
    (installed / "bin").mkdir(parents=True)
    record = job(tmp_path, python("raise SystemExit(5)")).model_dump_json()
    with pytest.raises(SystemExit) as ended:
        build(tmp_path)(["job", record])
    assert ended.value.code == 5


def test_the_package_runs_as_a_module_for_a_job_that_calls_its_own_tool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["mainboard", "--version"])
    with pytest.raises(SystemExit) as ended:
        runpy.run_module("mainboard", run_name="__main__")
    assert ended.value.code == 0
