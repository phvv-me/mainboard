from typing import TYPE_CHECKING, Annotated

import pytest
from pydantic import ValidationError

from mainboard.lab import Experiment, Fixed, IntRange, Lane, Run, experiment
from mainboard.lab.domains import space_of

if TYPE_CHECKING:
    from pathlib import Path


@experiment(models=("gpt2",), trials=3, seed=7)
def lane_aware(
    run: Run,
    *,
    bits: Annotated[int, IntRange(1, 8)] = 4,
    lane: Lane | None = None,
) -> dict[str, float]:
    return {"bits": float(bits), "has_lane": float(lane is not None)}


@experiment()
def lane_unaware(run: Run, *, bits: Annotated[int, Fixed(4)] = 4) -> dict[str, float]:
    return {"bits": float(bits)}


@experiment()
def with_required(
    run: Run,
    *,
    required: Annotated[int, Fixed(1)],
    optional: Annotated[int, IntRange(0, 10)] = 5,
) -> dict[str, float]:
    return {"required": float(required), "optional": float(optional)}


def test_experiment_decorator_registers_a_subclass_carrying_declarations() -> None:
    assert issubclass(lane_aware, Experiment)
    assert lane_aware.models == ("gpt2",)
    assert lane_aware.trials == 3
    assert lane_aware.seed == 7


def test_experiment_decorator_defaults_declarations_when_omitted() -> None:
    assert lane_unaware.models == ()
    assert lane_unaware.trials == 1
    assert lane_unaware.seed == 0
    assert lane_unaware.gates == ()
    assert lane_unaware.lanes == ()


def test_experiment_decorator_preserves_domain_metadata_for_space_of() -> None:
    assert space_of(lane_aware) == {"bits": IntRange(1, 8)}


def test_experiment_decorator_config_field_keeps_its_default() -> None:
    assert lane_unaware().bits == 4


def test_experiment_decorator_measure_forwards_lane_when_the_function_declares_it(
    tmp_path: Path,
) -> None:
    instance = lane_aware(bits=7)
    run = Run(model_id="gpt2", config=instance, artifact_dir=tmp_path)
    assert instance.measure(run, Lane(name="cold")) == {"bits": 7.0, "has_lane": 1.0}


def test_experiment_decorator_measure_omits_lane_when_the_function_does_not_declare_it(
    tmp_path: Path,
) -> None:
    instance = lane_unaware()
    run = Run(model_id="gpt2", config=instance, artifact_dir=tmp_path)
    assert instance.measure(run, Lane(name="cold")) == {"bits": 4.0}


def test_experiment_decorator_required_field_has_no_default() -> None:
    with pytest.raises(ValidationError):
        with_required()


def test_experiment_decorator_required_field_accepts_an_explicit_value(tmp_path: Path) -> None:
    instance = with_required(required=2)
    assert instance.optional == 5
    run = Run(model_id="gpt2", config=instance, artifact_dir=tmp_path)
    assert instance.measure(run) == {"required": 2.0, "optional": 5.0}


def test_decorated_experiments_register_under_kebab_names() -> None:
    @experiment()
    def multi_word_probe(run) -> dict[str, float]:
        """A probe whose function name is snake case."""
        return {"value": 1.0}

    assert multi_word_probe.name == "multi-word-probe"
    assert Experiment.find("multi-word-probe") is multi_word_probe
