import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path

import pytest
from pydantic import JsonValue

from mainboard import Project
from mainboard.lab import Experiment, Lane, Run
from mainboard.lab.board_surface import (
    RECEIPT,
    BlockedTrial,
    FailedTrial,
    TrialOutcome,
    TrialResult,
    runnable,
)
from mainboard.lab.gates import Gate, GateStatus, GateVerdict

# The one gate entry every mixed sweep below renders first, kept here so each expected receipt
# spells out only the gate that decides its outcome.
_PASSED_GATE = {"status": "passed", "reason": ""}


@dataclass(frozen=True, slots=True)
class FixedGate(Gate):
    """A gate answering a predetermined verdict, so the sweep around it is what gets measured.

    outcome: the status this gate always reports.
    reason: the explanation it carries alongside.
    """

    outcome: GateStatus
    reason: str = ""

    def check(self, context: Run) -> GateVerdict:
        return GateVerdict(status=self.outcome, reason=self.reason)


class NoGates(Experiment):
    """A trial with no preconditions at all."""

    def measure(self, run: Run, lane: Lane | None = None) -> dict[str, float]:
        return {"score": 1.0}


class MixedPassAndBlock(NoGates):
    """A trial whose sweep clears one gate and is withheld by the next."""

    gates = (FixedGate(GateStatus.PASSED), FixedGate(GateStatus.BLOCKED, reason="wait"))


class MixedPassAndFail(NoGates):
    """A trial whose sweep clears one gate and breaks on the next."""

    gates = (FixedGate(GateStatus.PASSED), FixedGate(GateStatus.FAILED, reason="broke"))


class BlockedOutranksFailed(NoGates):
    """A trial blocked by one gate and failed by another, the precedence case."""

    gates = (
        FixedGate(GateStatus.BLOCKED, reason="wait"),
        FixedGate(GateStatus.FAILED, reason="broke"),
    )


@pytest.mark.parametrize(
    ("declared", "outcome_kind", "reason", "swept"),
    [
        pytest.param(NoGates, TrialResult, "", (), id="no-gates-runs-to-completion"),
        pytest.param(
            MixedPassAndBlock,
            BlockedTrial,
            "wait",
            (GateStatus.PASSED, GateStatus.BLOCKED),
            id="a-blocked-gate-withholds-it",
        ),
        pytest.param(
            MixedPassAndFail,
            FailedTrial,
            "broke",
            (GateStatus.PASSED, GateStatus.FAILED),
            id="a-broken-gate-fails-it",
        ),
        pytest.param(
            BlockedOutranksFailed,
            BlockedTrial,
            "wait",
            (GateStatus.BLOCKED, GateStatus.FAILED),
            id="a-block-outranks-a-failure",
        ),
    ],
)
def test_runnable_sweeps_every_declared_gate_before_reporting_the_trials_outcome(
    declared: type[Experiment],
    outcome_kind: type[TrialOutcome],
    reason: str,
    swept: tuple[GateStatus, ...],
) -> None:
    outcome = runnable(declared, "gpt2", declared())
    assert isinstance(outcome, outcome_kind)
    assert outcome.run_id == declared().run_id(model="gpt2")
    assert tuple(verdict.status for verdict in outcome.gate_evidence) == swept
    assert getattr(outcome, "reason", "") == reason


def test_runnable_builds_the_artifact_dir_under_the_projects_runs_path() -> None:
    captured: list[Path] = []

    class Capturing(Experiment):
        """An experiment that reports back the artifact dir its trial was handed."""

        def measure(self, run: Run, lane: Lane | None = None) -> dict[str, float]:
            captured.append(run.artifact_dir)
            return {}

    outcome = runnable(Capturing, "gpt2", Capturing())
    assert captured == [Path(Project().out_dir) / "runs" / outcome.run_id]


@pytest.mark.parametrize(
    ("declared", "word", "gates", "own", "absent"),
    [
        pytest.param(
            NoGates, "passed", [], {"metrics": {"score": 1.0}}, "reason", id="a-completed-trial"
        ),
        pytest.param(
            MixedPassAndBlock,
            "blocked",
            [_PASSED_GATE, {"status": "blocked", "reason": "wait"}],
            {"reason": "wait"},
            "metrics",
            id="a-withheld-trial",
        ),
        pytest.param(
            MixedPassAndFail,
            "failed",
            [_PASSED_GATE, {"status": "failed", "reason": "broke"}],
            {"reason": "broke"},
            "metrics",
            id="a-broken-trial",
        ),
    ],
)
def test_every_outcome_renders_one_receipt_line_under_the_one_published_key(
    declared: type[Experiment],
    word: str,
    gates: list[dict[str, str]],
    own: Mapping[str, JsonValue],
    absent: str,
) -> None:
    outcome = runnable(declared, "gpt2", declared())
    line = outcome.receipt()
    assert "\n" not in line
    assert set(json.loads(line)) == {RECEIPT}
    record = json.loads(line)[RECEIPT]
    assert record["run_id"] == outcome.run_id
    assert record["outcome"] == word
    assert record["producer"] == Project().name
    assert record["gates"] == gates
    assert absent not in record
    for field, value in own.items():
        assert record[field] == value
    # The node field is optional both ways: absent from an undeclared receipt, printed when a
    # trial names the ledger node it serves.
    assert "node" not in record
    named = replace(outcome, node="invariance-tax-law")
    assert json.loads(named.receipt())[RECEIPT]["node"] == "invariance-tax-law"
