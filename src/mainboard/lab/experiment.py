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
    """The frozen config-fields base of every `Experiment`.

    Kept apart so `Run.config` is typed against the trial's validated fields without the
    registry, gates and lane machinery.
    """


class Experiment(Registry, DeclaredExperiment, abc.ABC):
    """A declared experiment: its config fields, its trial preconditions, and how to measure.

    A subclass adds pydantic fields for the swept config, each `Annotated` with a domain marker,
    and overrides `measure`. `lanes` are counterbalanced by `lab.lane.orders`, `gates` checked by
    `runnable` before `setup` and `measure`.

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
        """Measure this trial and return its named metrics; `lane` is None without lanes."""

    def run_id(self, *, model: str, lane: Lane | None = None) -> str:
        """This trial's dedup identity: the config plus the model and lane name."""
        lane_name = lane.name if lane is not None else ""
        return content_hash({**self.model_dump(mode="json"), "model": model, "lane": lane_name})

    def setup(self, run: Run) -> Fixture:
        """Prepare this trial's shared fixture; the default does nothing."""
        return None
