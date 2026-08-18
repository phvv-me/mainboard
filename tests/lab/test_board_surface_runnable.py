from dataclasses import dataclass
from pathlib import Path

from mainboard import Project
from mainboard.lab import Experiment, Lane, Run
from mainboard.lab.board_surface import BlockedTrial, FailedTrial, TrialResult, runnable
from mainboard.lab.gates import Gate, GateStatus, GateVerdict


@dataclass(frozen=True, slots=True)
class FixedGate(Gate):
    outcome: GateStatus
    reason: str = ""

    def check(self, context: Run) -> GateVerdict:
        return GateVerdict(status=self.outcome, reason=self.reason)


class NoGates(Experiment):
    def measure(self, run: Run, lane: Lane | None = None) -> dict[str, float]:
        return {"score": 1.0}


class MixedPassAndBlock(Experiment):
    gates = (FixedGate(GateStatus.PASSED), FixedGate(GateStatus.BLOCKED, reason="wait"))

    def measure(self, run: Run, lane: Lane | None = None) -> dict[str, float]:
        return {"score": 1.0}


class MixedPassAndFail(Experiment):
    gates = (FixedGate(GateStatus.PASSED), FixedGate(GateStatus.FAILED, reason="broke"))

    def measure(self, run: Run, lane: Lane | None = None) -> dict[str, float]:
        return {"score": 1.0}


class BlockedOutranksFailed(Experiment):
    gates = (
        FixedGate(GateStatus.BLOCKED, reason="wait"),
        FixedGate(GateStatus.FAILED, reason="broke"),
    )

    def measure(self, run: Run, lane: Lane | None = None) -> dict[str, float]:
        return {"score": 1.0}


def test_runnable_returns_trial_result_when_no_gates_are_declared() -> None:
    outcome = runnable(NoGates, "gpt2", NoGates())
    assert isinstance(outcome, TrialResult)
    assert outcome.metrics == {"score": 1.0}
    assert outcome.gate_evidence == ()


def test_runnable_returns_blocked_trial_and_keeps_every_gates_evidence() -> None:
    outcome = runnable(MixedPassAndBlock, "gpt2", MixedPassAndBlock())
    assert isinstance(outcome, BlockedTrial)
    assert outcome.reason == "wait"
    assert [verdict.status for verdict in outcome.gate_evidence] == [
        GateStatus.PASSED,
        GateStatus.BLOCKED,
    ]


def test_runnable_returns_failed_trial_when_no_gate_is_blocked() -> None:
    outcome = runnable(MixedPassAndFail, "gpt2", MixedPassAndFail())
    assert isinstance(outcome, FailedTrial)
    assert outcome.reason == "broke"


def test_runnable_blocked_outranks_failed() -> None:
    outcome = runnable(BlockedOutranksFailed, "gpt2", BlockedOutranksFailed())
    assert isinstance(outcome, BlockedTrial)
    assert outcome.reason == "wait"


def test_runnable_builds_the_artifact_dir_under_the_project_runs_path() -> None:
    captured: dict[str, Path] = {}

    class Capturing(Experiment):
        def measure(self, run: Run, lane: Lane | None = None) -> dict[str, float]:
            captured["artifact_dir"] = run.artifact_dir
            return {}

    instance = Capturing()
    outcome = runnable(Capturing, "gpt2", instance)
    assert captured["artifact_dir"] == Path(Project().out_dir) / "runs" / outcome.run_id
