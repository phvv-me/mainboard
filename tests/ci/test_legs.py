import base64
import subprocess
from pathlib import Path

import pytest

from mainboard import MissionError
from mainboard.ci import SHIP, LocalLeg, Package, RemoteLeg, Result, Step, Verdict
from mainboard.context.plan import ExecutionPlan
from mainboard.dispatch.transport import HostUnreachable, SshTransport

from .conftest import PYTHON, Ssh, declare, plan, say, step


def _verdicts(results: list[Result]) -> list[tuple[str, Verdict]]:
    return [(result.step, result.verdict) for result in results]


def test_a_local_leg_stops_at_the_first_failure_and_names_what_it_never_ran(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each step runs in the package with its own output; the inherited venv stays behind."""
    monkeypatch.setenv("VIRTUAL_ENV", "/elsewhere")
    env = f"{PYTHON} -c \"import os; print(os.getcwd(), os.environ.get('VIRTUAL_ENV'))\""
    gate = step("where", env) + step("fail", say("broken", 3)) + step("after", say("never"))
    package = Package.found(declare(tmp_path / "pkg", gate))

    leg = LocalLeg(package.root)
    results = list(leg.run(package.definition.on(leg.family)))

    assert _verdicts(results) == [
        ("where", Verdict.OK),
        ("fail", Verdict.FAILED),
        ("after", Verdict.NOT_RUN),
    ]
    where, venv = results[0].output.split()
    assert (Path(where).resolve(), venv) == (package.root.resolve(), "None")
    assert results[1].transcript.startswith(f"local [{leg.family}] fail: failed in ")
    assert results[1].transcript.endswith("broken")
    assert results[2].transcript == ""
    assert results[1].row() == {
        "leg": "local",
        "os": leg.family,
        "step": "fail",
        "verdict": Verdict.FAILED,
        "seconds": round(results[1].seconds, 1),
    }
    assert [result.failed for result in results] == [False, True, False]


@pytest.mark.parametrize(
    ("run", "timeout", "verdict"),
    [
        ("no-such-program-anywhere --flag", 5.0, Verdict.MISSING),
        (f'{PYTHON} -c "import time; time.sleep(30)"', 0.3, Verdict.TIMED_OUT),
    ],
    ids=["a program that is not there", "a step past its deadline"],
)
def test_a_step_that_could_not_finish_says_why(
    tmp_path: Path, run: str, timeout: float, verdict: Verdict
) -> None:
    (result,) = LocalLeg(tmp_path).run([Step(name="s", run=run, timeout=timeout)])
    assert result.verdict is verdict


def _shipped(calls: list[tuple[ExecutionPlan, str, str]]):
    def ship(target: ExecutionPlan, root: str, package: str) -> None:
        calls.append((target, root, package))

    return ship


def test_a_windows_leg_ships_first_then_runs_each_step_in_powershell_from_its_own_copy(
    ssh: Ssh,
) -> None:
    ssh.answers.extend([(0, "clean\n", ""), (1, "", '#< CLIXML\n<S S="Error">boom</S>')])
    shipped: list[tuple[ExecutionPlan, str, str]] = []
    leg = RemoteLeg(plan("box", "win-64"), "packages/p", ship=_shipped(shipped), ssh=ssh)
    gate = [Step(name="lint", run="uv run ruff check ."), Step(name="test", run="uv run pytest")]

    results = list(leg.run(gate))

    assert (leg.name, leg.family) == ("box", "win")
    assert shipped == [(leg.plan, "/m/.mainboard/ci", "packages/p")]
    assert _verdicts(results) == [
        (SHIP, Verdict.OK),
        ("lint", Verdict.OK),
        ("test", Verdict.FAILED),
    ]
    assert results[2].output == "boom"
    command, timeout = ssh.calls[0]
    assert timeout == gate[0].timeout
    assert command[:1] == ("ssh",) and "box" in command
    script = base64.b64decode(command[-1]).decode("utf-16-le")
    assert "Set-Location -LiteralPath '/m/.mainboard/ci/packages/p'" in script
    assert "& 'uv' 'run' 'ruff' 'check' '.'" in script


def test_a_posix_leg_runs_each_step_under_a_login_bash_in_its_own_copy(ssh: Ssh) -> None:
    ssh.answers.append((0, "ok\n", ""))
    leg = RemoteLeg(plan("gpu", "linux-64"), "packages/p", ship=_shipped([]), ssh=ssh)

    list(leg.run([Step(name="test", run="uv run pytest -k 'a b'")]))

    command, _ = ssh.calls[0]
    assert command[-2] == "gpu"
    assert command[-1].startswith("bash -lc ")
    assert "cd /m/.mainboard/ci/packages/p" in command[-1]
    assert "uv run pytest -k '\"'\"'a b'\"'\"'" in command[-1]


def test_a_host_that_never_answers_reads_as_unreachable_and_a_hung_step_as_timed_out(
    ssh: Ssh,
) -> None:
    hung = HostUnreachable("ssh ci test to 'gpu' timed out after 1s")
    hung.__cause__ = subprocess.TimeoutExpired("ssh", 1.0)
    ssh.answers.extend([HostUnreachable("ssh ci lint to 'gpu' failed: refused"), hung])
    leg = RemoteLeg(plan("gpu", "linux-64"), "p", ship=_shipped([]), ssh=ssh)

    (lint,) = [result for result in leg.run([Step(name="lint", run="x")]) if result.step != SHIP]
    (test,) = [result for result in leg.run([Step(name="test", run="x")]) if result.step != SHIP]

    assert (lint.verdict, test.verdict) == (Verdict.UNREACHABLE, Verdict.TIMED_OUT)
    assert "refused" in lint.output


def test_a_package_that_never_arrived_runs_nothing(ssh: Ssh) -> None:
    def refuse(target: ExecutionPlan, root: str, package: str) -> None:
        raise HostUnreachable("mirror to 'gpu' failed: no route")

    leg = RemoteLeg(plan("gpu", "linux-64"), "p", ship=refuse, ssh=ssh)
    results = list(leg.run([Step(name="lint", run="x"), Step(name="test", run="y")]))

    assert _verdicts(results) == [
        (SHIP, Verdict.UNREACHABLE),
        ("lint", Verdict.NOT_RUN),
        ("test", Verdict.NOT_RUN),
    ]
    assert "no route" in results[0].output
    assert ssh.calls == []


def test_a_leg_needs_its_hosts_platform_and_rides_the_default_ssh_policy_unless_given_one() -> (
    None
):
    with pytest.raises(MissionError, match=r"declare \[hosts.gpu\] platform"):
        RemoteLeg(plan("gpu", ""), "p", ship=_shipped([]))
    leg = RemoteLeg(plan("gpu", "osx-arm64"), "p", ship=_shipped([]))
    assert (leg.family, leg.ssh) == ("osx", SshTransport())
