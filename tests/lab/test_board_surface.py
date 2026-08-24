from typing import Annotated

import pytest
from pydantic import ValidationError

from mainboard.lab import Experiment, Fixed, IntRange, Lane, Run, experiment
from mainboard.lab.domains import space_of


@experiment(models=("gpt2",), trials=3, seed=7)
def lane_aware(
    run: Run,
    *,
    bits: Annotated[int, IntRange(1, 8)] = 4,
    lane: Lane | None = None,
) -> dict[str, float]:
    """A measuring function that asks for the counterbalanced lane itself."""
    return {"bits": float(bits), "has_lane": float(lane is not None)}


@experiment()
def lane_unaware(run: Run, *, bits: Annotated[int, Fixed(4)] = 4) -> dict[str, float]:
    """A measuring function that never declares a lane parameter."""
    return {"bits": float(bits)}


@experiment()
def with_required(
    run: Run,
    *,
    required: Annotated[int, Fixed(1)],
    optional: Annotated[int, IntRange(0, 10)] = 5,
) -> dict[str, float]:
    """A measuring function mixing a parameter with no default and one that has one."""
    return {"required": float(required), "optional": float(optional)}


@pytest.mark.parametrize(
    ("declared", "models", "trials", "seed"),
    [
        pytest.param(lane_aware, ("gpt2",), 3, 7, id="declarations-carried-through"),
        pytest.param(lane_unaware, (), 1, 0, id="declarations-left-at-their-defaults"),
    ],
)
def test_the_decorator_builds_a_registered_subclass_carrying_its_declarations(
    declared: type[Experiment], models: tuple[str, ...], trials: int, seed: int
) -> None:
    assert issubclass(declared, Experiment)
    assert (declared.models, declared.trials, declared.seed) == (models, trials, seed)
    assert (declared.gates, declared.lanes) == ((), ())


def test_the_decorator_carries_each_config_parameters_domain_through_to_space_of() -> None:
    assert space_of(lane_aware) == {"bits": IntRange(1, 8)}
    assert space_of(with_required) == {"required": Fixed(1), "optional": IntRange(0, 10)}


def test_a_config_parameter_keeps_its_default_and_one_without_a_default_stays_required() -> None:
    assert lane_unaware().bits == 4
    assert with_required(required=2).optional == 5
    with pytest.raises(ValidationError):
        with_required()


@pytest.mark.parametrize(
    ("config", "metrics"),
    [
        pytest.param(
            lane_aware(bits=7), {"bits": 7.0, "has_lane": 1.0}, id="the-lane-reaches-a-declarer"
        ),
        pytest.param(lane_unaware(), {"bits": 4.0}, id="a-non-declarer-never-sees-the-lane"),
        pytest.param(
            with_required(required=2),
            {"required": 2.0, "optional": 5.0},
            id="every-config-field-is-forwarded",
        ),
    ],
)
def test_measure_forwards_every_config_field_and_the_lane_only_to_a_function_declaring_it(
    context: Run, config: Experiment, metrics: dict[str, float]
) -> None:
    assert config.measure(context, Lane(name="cold")) == metrics


def test_decorated_experiments_register_under_kebab_names() -> None:
    """The generated class declares its own registry name.

    A function's snake_case would leak into the registry key, where every other
    implementation is kebab.
    """

    @experiment()
    def multi_word_probe(run: Run) -> dict[str, float]:
        """A probe whose function name is snake case."""
        return {"value": 1.0}

    assert multi_word_probe.name == "multi-word-probe"
    assert Experiment.find("multi-word-probe") is multi_word_probe
