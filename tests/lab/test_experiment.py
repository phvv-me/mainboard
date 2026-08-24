from hypothesis import given
from hypothesis import strategies as st

from mainboard.lab import Experiment, Lane, Run
from mainboard.lab.experiment import DeclaredExperiment

from ..strategies import WORDS


class Minimal(Experiment):
    """The smallest declared experiment, no config fields and nothing declared around them."""

    def measure(self, run: Run, lane: Lane | None = None) -> dict[str, float]:
        return {"seen": 1.0}


def test_an_experiment_declares_nothing_by_default_and_its_setup_does_nothing(
    context: Run,
) -> None:
    assert (Minimal.lanes, Minimal.gates, Minimal.models) == ((), (), ())
    assert (Minimal.trials, Minimal.seed) == (1, 0)
    assert issubclass(Minimal, DeclaredExperiment)
    assert Minimal().setup(context) is None


@given(
    models=st.lists(WORDS, unique=True, min_size=2, max_size=3),
    lanes=st.lists(WORDS, unique=True, min_size=1, max_size=2),
)
def test_a_trial_identity_is_stable_for_one_config_and_distinct_per_model_and_lane(
    *, models: list[str], lanes: list[str]
) -> None:
    instance = Minimal()

    def identify(model: str, name: str) -> str:
        return instance.run_id(model=model, lane=Lane(name=name) if name else None)

    # The empty lane name stands for the undeclared lane, which is the identity `run_id` folds
    # a missing lane into, so it belongs in the same distinctness check as the named ones.
    identities = {
        (model, name): identify(model, name) for model in models for name in ["", *lanes]
    }
    assert len(set(identities.values())) == len(identities)
    assert all(identify(*trial) == identity for trial, identity in identities.items())
