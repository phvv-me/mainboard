from typing import TYPE_CHECKING, NoReturn

import pytest

from mainboard import HfDataset, wait_for_idle
from mainboard.lab import Idle, Offline, Parity, Receipt, Run
from mainboard.lab.experiment import DeclaredExperiment
from mainboard.lab.gates import GateStatus, is_offline_declared, is_parity_assumed

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def context(tmp_path: Path) -> Run:
    return Run(model_id="gpt2", config=DeclaredExperiment(), artifact_dir=tmp_path)


def raise_boom() -> NoReturn:
    raise RuntimeError("boom")


def test_idle_check_passes_when_wait_reports_idle(context: Run) -> None:
    verdict = Idle(seconds=1.0, wait=lambda **kw: True).check(context)
    assert verdict.status == GateStatus.PASSED
    assert not verdict.reason


def test_idle_check_blocks_when_wait_times_out(context: Run) -> None:
    verdict = Idle(seconds=2.5, wait=lambda **kw: False).check(context)
    assert verdict.status == GateStatus.BLOCKED
    assert "2.5" in verdict.reason


def test_idle_check_fails_when_wait_raises(context: Run) -> None:
    verdict = Idle(seconds=1.0, wait=lambda **kw: raise_boom()).check(context)
    assert verdict.status == GateStatus.FAILED
    assert verdict.reason == "boom"


def test_idle_defaults_to_mainboard_wait_for_idle() -> None:
    assert Idle(seconds=1.0).wait is wait_for_idle


def test_is_parity_assumed_is_permissive(context: Run) -> None:
    assert is_parity_assumed("oracle", context) is True


def test_parity_check_passes_with_default_probe(context: Run) -> None:
    verdict = Parity(oracle="reference").check(context)
    assert verdict.status == GateStatus.PASSED


def test_parity_check_blocks_when_probe_reports_mismatch(context: Run) -> None:
    verdict = Parity(oracle="reference", probe=lambda oracle, ctx: False).check(context)
    assert verdict.status == GateStatus.BLOCKED
    assert "reference" in verdict.reason


def test_parity_check_fails_when_probe_raises(context: Run) -> None:
    verdict = Parity(oracle="reference", probe=lambda oracle, ctx: raise_boom()).check(context)
    assert verdict.status == GateStatus.FAILED
    assert verdict.reason == "boom"


def test_is_offline_declared_reads_the_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    assert is_offline_declared() is True
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    assert is_offline_declared() is False


def test_offline_check_passes_when_probe_reports_offline(context: Run) -> None:
    verdict = Offline(probe=lambda: True).check(context)
    assert verdict.status == GateStatus.PASSED


def test_offline_check_blocks_when_probe_reports_online(context: Run) -> None:
    verdict = Offline(probe=lambda: False).check(context)
    assert verdict.status == GateStatus.BLOCKED


def test_offline_check_fails_when_probe_raises(context: Run) -> None:
    verdict = Offline(probe=raise_boom).check(context)
    assert verdict.status == GateStatus.FAILED
    assert verdict.reason == "boom"


def test_receipt_check_passes_when_nothing_is_declared(context: Run) -> None:
    verdict = Receipt(dataset="org/data@main").check(context)
    assert verdict.status == GateStatus.PASSED


def test_receipt_check_blocks_when_declared_dataset_is_missing(context: Run) -> None:
    gate = Receipt(dataset="org/data@main", needs=(HfDataset(repo="org/data"),), staged=tuple)
    verdict = gate.check(context)
    assert verdict.status == GateStatus.BLOCKED
    assert "org/data@main" in verdict.reason


def test_receipt_check_passes_when_declared_dataset_is_staged(context: Run) -> None:
    gate = Receipt(
        dataset="org/data@main",
        needs=(HfDataset(repo="org/data"),),
        staged=lambda: ("org/data@main",),
    )
    verdict = gate.check(context)
    assert verdict.status == GateStatus.PASSED


def test_receipt_check_fails_when_staged_lookup_raises(context: Run) -> None:
    verdict = Receipt(dataset="org/data@main", staged=raise_boom).check(context)
    assert verdict.status == GateStatus.FAILED
    assert verdict.reason == "boom"
