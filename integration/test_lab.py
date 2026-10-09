"""The lab vocabulary an experiment declares itself with: markers beside pydantic fields, gates
and the receipt a trial prints."""

import json
from pathlib import Path
from typing import Annotated

import pytest
from pydantic import ValidationError

from mainboard.lab import Choices, Experiment, Fixed, FloatRange, IntRange, Offline
from mainboard.lab.board_surface import RECEIPT, runnable
from mainboard.lab.domains import space_of


class Sweep(Experiment):
    """An experiment with one of each marker, written positionally as researchers write them."""

    width: Annotated[int, IntRange(1, 8)] = 2
    mode: Annotated[str, Choices("a", "b")] = "a"
    rate: Annotated[float, FloatRange(0.0, 1.0)] = 0.5
    codec: Annotated[str, Fixed("e8")] = "e8"

    def measure(self, run, lane=None) -> dict[str, float]:
        return {"width": float(self.width)}


class Offstage(Sweep):
    gates = (Offline(probe=lambda: False),)


def test_markers_annotate_a_field_without_replacing_its_schema() -> None:
    assert Sweep(width=5).width == 5
    with pytest.raises(ValidationError):
        Sweep(width="wide")
    assert space_of(Sweep) == {
        "width": IntRange(1, 8),
        "mode": Choices("a", "b"),
        "rate": FloatRange(0.0, 1.0),
        "codec": Fixed("e8"),
    }


def test_a_trial_prints_one_receipt_whether_it_ran_or_a_gate_held_it(
    workspace: Path, monkeypatch
) -> None:
    monkeypatch.chdir(workspace)
    ran = json.loads(runnable(Sweep, "m", Sweep(width=3)).receipt())[RECEIPT]
    held = json.loads(runnable(Offstage, "m", Offstage()).receipt())[RECEIPT]

    assert (ran["outcome"], ran["metrics"], ran["gates"]) == ("passed", {"width": 3.0}, [])
    assert (held["outcome"], held["reason"]) == ("blocked", "offline mode is not declared")
    assert held["gates"] == [{"status": "blocked", "reason": held["reason"]}]
