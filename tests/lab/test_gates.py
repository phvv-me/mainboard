from collections.abc import Callable
from typing import NoReturn

import pytest

from mainboard import HfDataset, wait_for_idle
from mainboard.lab import Idle, Offline, Parity, Receipt, Run
from mainboard.lab.gates import (
    Gate,
    GateStatus,
    GateVerdict,
    is_offline_declared,
    is_parity_assumed,
)


def is_met() -> bool:
    """A probe reporting its precondition already holds."""
    return True


def is_unmet() -> bool:
    """A probe cleanly reporting its precondition does not hold yet."""
    return False


def broken() -> NoReturn:
    """A probe whose own check breaks rather than answering."""
    raise RuntimeError("boom")


def idle_gate(answer: Callable[[], bool]) -> Gate:
    """An idle gate whose wait defers to `answer`."""
    return Idle(seconds=2.5, wait=lambda **_: answer())


def parity_gate(answer: Callable[[], bool]) -> Gate:
    """A parity gate whose comparison probe defers to `answer`."""
    return Parity(oracle="reference", probe=lambda oracle, context: answer())


def offline_gate(answer: Callable[[], bool]) -> Gate:
    """An offline gate whose declaration probe is `answer` itself."""
    return Offline(probe=answer)


def receipt_gate(answer: Callable[[], bool]) -> Gate:
    """A receipt gate whose staging lookup reports the declared dataset when `answer` says so."""
    return Receipt(
        dataset="org/data@main",
        needs=(HfDataset(repo="org/data"),),
        staged=lambda: ("org/data@main",) if answer() else (),
    )


@pytest.mark.parametrize(
    ("build", "blocked"),
    [
        pytest.param(idle_gate, "GPU still busy after 2.5s", id="idle"),
        pytest.param(parity_gate, "parity with 'reference' not established", id="parity"),
        pytest.param(offline_gate, "offline mode is not declared", id="offline"),
        pytest.param(receipt_gate, "org/data@main is not staged yet", id="receipt"),
    ],
)
def test_every_gate_translates_its_own_probe_into_the_shared_three_way_verdict(
    context: Run, build: Callable[[Callable[[], bool]], Gate], blocked: str
) -> None:
    assert build(is_met).check(context) == GateVerdict(status=GateStatus.PASSED)
    assert build(is_unmet).check(context) == GateVerdict(status=GateStatus.BLOCKED, reason=blocked)
    assert build(broken).check(context) == GateVerdict(status=GateStatus.FAILED, reason="boom")


def test_the_hardware_free_defaults_assume_parity_stage_nothing_and_read_offline_off_the_env(
    context: Run, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert Idle(seconds=1.0).wait is wait_for_idle
    assert Parity(oracle="reference").probe is is_parity_assumed
    assert is_parity_assumed("reference", context) is True
    assert Parity(oracle="reference").check(context).status is GateStatus.PASSED
    assert Receipt(dataset="org/data@main").check(context).status is GateStatus.PASSED
    assert Offline().probe is is_offline_declared
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    assert is_offline_declared() is True
    monkeypatch.delenv("HF_HUB_OFFLINE")
    assert is_offline_declared() is False
