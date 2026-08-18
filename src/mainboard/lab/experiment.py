import abc
from typing import TYPE_CHECKING, ClassVar, Protocol

from patos import FrozenModel, Registry

from ..experiments.identity import run_id as content_hash

if TYPE_CHECKING:
    from .gates import Gate
    from .lane import Lane
    from .run import Run


class Fixture(Protocol):
    """An intentionally unconstrained value `setup` hands on to `measure`."""


class DeclaredExperiment(FrozenModel):
    """The frozen config-fields base every concrete `Experiment` subclass extends.

    Kept apart from `Experiment` itself so `Run.config` can be typed against a plain frozen
    model, the trial's validated fields, without pulling in the registry, gates, and lane
    machinery only a fully declared experiment needs.
    """


class Experiment(Registry, DeclaredExperiment, abc.ABC):
    """A declared experiment: its config fields, its trial preconditions, and how to measure.

    A subclass adds its own pydantic fields for the swept config, each `Annotated` with a
    `Choices`/`IntRange`/`FloatRange`/`Fixed` domain, and overrides `measure`. Declaring
    `lanes` hands their counterbalancing entirely to the framework (see `lab.lane.orders`);
    declaring `gates` hands their precondition checks entirely to `runnable`.

    lanes: the counterbalanced conditions each trial's `measure` runs under.
    gates: the preconditions `runnable` checks before `setup` and `measure`.
    models: the model ids this experiment sweeps.
    trials: how many trial blocks a study runs.
    seed: the base random seed a study derives per-trial seeds from.
    """

    lanes: ClassVar[tuple[Lane, ...]] = ()
    gates: ClassVar[tuple[Gate, ...]] = ()
    models: ClassVar[tuple[str, ...]] = ()
    trials: ClassVar[int] = 1
    seed: ClassVar[int] = 0

    @abc.abstractmethod
    def measure(self, run: Run, lane: Lane | None = None) -> dict[str, float]:
        """Measure this trial and return its named metrics.

        run: this trial's context.
        lane: the counterbalanced condition under measurement, None when the experiment
            declares no lanes.
        """

    def run_id(self, *, model: str, lane: Lane | None = None) -> str:
        """This trial's dedup identity: the config plus the model and lane name.

        model: the model id this trial runs against.
        lane: the counterbalanced condition this trial measures, when any is declared.
        """
        payload = {
            **self.model_dump(mode="json"),
            "model": model,
            "lane": lane.name if lane is not None else "",
        }
        return content_hash(payload)

    def setup(self, run: Run) -> Fixture:
        """Prepare this trial's shared fixture; the framework default does nothing.

        run: this trial's context, in case setup needs a scratch path or the resolved model id.
        """
        return None
