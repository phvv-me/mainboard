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
from mainboard.lab.gates import Gate, GateStatus, GateVerdict, Idle, Parity

# The one gate entry every mixed sweep below renders first, kept here so each expected receipt
# spells out only the gate that decides its outcome.
_PASSED_GATE = {"status": "passed", "reason": ""}


@dataclass(frozen=True, slots=True)
class FixedGate(Gate):
    """A gate answering a predetermined verdict, so the sweep around it is what gets measured."""

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


class BlockedBeforeFailed(NoGates):
    """A trial whose first blocker prevents checking its later failing gate."""

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
            BlockedBeforeFailed,
            BlockedTrial,
            "wait",
            (GateStatus.BLOCKED,),
            id="a-block-prevents-later-checks",
        ),
    ],
)
def test_runnable_retains_only_evaluated_gates_in_declaration_order(
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


@pytest.mark.parametrize("status", tuple(GateStatus))
def test_idle_admits_parity_and_measurement_only_after_passing(
    monkeypatch: pytest.MonkeyPatch, status: GateStatus
) -> None:
    calls: list[str] = []

    def wait(*, timeout: float) -> bool:
        assert timeout == 0
        calls.append("idle")
        if status == GateStatus.FAILED:
            raise RuntimeError("idle probe failed")
        return status == GateStatus.PASSED

    def parity(oracle: str, run: Run) -> bool:
        calls.append("parity")
        return True

    def setup(self: NoGates, run: Run) -> None:
        calls.append("setup")

    def measure(self: NoGates, run: Run, lane: Lane | None = None) -> dict[str, float]:
        calls.append("measure")
        return {"score": 1.0}

    monkeypatch.setattr(NoGates, "gates", (Idle(seconds=0, wait=wait), Parity("hf", parity)))
    monkeypatch.setattr(NoGates, "setup", setup)
    monkeypatch.setattr(NoGates, "measure", measure)
    outcome = runnable(NoGates, "gpt2", NoGates())

    assert outcome.verdict == status
    expected = ["idle", "parity", "setup", "measure"] if status == GateStatus.PASSED else ["idle"]
    assert calls == expected
    assert tuple(verdict.status for verdict in outcome.gate_evidence) == (
        (GateStatus.PASSED, GateStatus.PASSED) if status == GateStatus.PASSED else (status,)
    )


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
