from typing import TYPE_CHECKING

from mainboard.lab import Experiment, Lane, Run
from mainboard.lab.experiment import DeclaredExperiment

if TYPE_CHECKING:
    from pathlib import Path


class Minimal(Experiment):
    def measure(self, run: Run, lane: Lane | None = None) -> dict[str, float]:
        return {"seen": 1.0}


def test_experiment_classvar_defaults_are_empty_and_single_trial() -> None:
    assert Minimal.lanes == ()
    assert Minimal.gates == ()
    assert Minimal.models == ()
    assert Minimal.trials == 1
    assert Minimal.seed == 0


def test_experiment_is_a_declared_experiment() -> None:
    assert issubclass(Minimal, DeclaredExperiment)


def test_experiment_setup_default_does_nothing(tmp_path: Path) -> None:
    run = Run(model_id="gpt2", config=Minimal(), artifact_dir=tmp_path)
    assert Minimal().setup(run) is None


def test_run_id_is_stable_for_the_same_inputs() -> None:
    instance = Minimal()
    first = instance.run_id(model="gpt2")
    second = instance.run_id(model="gpt2")
    assert first == second


def test_run_id_varies_with_the_model() -> None:
    instance = Minimal()
    assert instance.run_id(model="gpt2") != instance.run_id(model="llama")


def test_run_id_varies_with_the_lane() -> None:
    instance = Minimal()
    without_lane = instance.run_id(model="gpt2")
    with_lane = instance.run_id(model="gpt2", lane=Lane(name="cold"))
    assert without_lane != with_lane


def test_run_id_varies_between_lanes() -> None:
    instance = Minimal()
    cold = instance.run_id(model="gpt2", lane=Lane(name="cold"))
    warm = instance.run_id(model="gpt2", lane=Lane(name="warm"))
    assert cold != warm
