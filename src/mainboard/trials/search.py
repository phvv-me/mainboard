# Search lanes: a worst case proposed adaptively, one receipt row per evaluation.
#
# A search lane spends a real evaluation per point, a GPU measurement returning a continuous
# misfit, where a grid at any useful resolution is unaffordable, so a sampler steered by the misfit
# finds the worst corner inside a budget. The ledger is the storage and the sampler is not: the
# driver is asked for the next point and told what it scored (ask-tell, which needs no storage, so
# none is opened), and every iteration is a receipt row through the same trial and store as a
# declared lane, riding samples-per-cell. The study's outcome is one more row on top, so marked.
#
# The lane settles on the study, not on a point: a worst point outside the law's band settles the
# consumer's refuted word naming that point, otherwise its survival word with the count stated. The
# whole budget is spent either way, since the object reported is the WORST point. That point is a
# candidate under the rule `adaptive` states. The loss is always maximised, since a search lane
# hunts a worst case; a consumer whose misfit reads the other way negates it.

from typing import TYPE_CHECKING, Protocol

from patos import FrozenModel

# At runtime, not under `TYPE_CHECKING`: pydantic resolves `Miss`'s field annotations when the
# class is built.
from pydantic import JsonValue

from .adaptive import Owed, driver

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from .session import Trial


class Miss(FrozenModel):
    """What one evaluated point of a search scored, and what was read to score it.

    loss: the continuous misfit, which steers the sampler and which the study maximises.
    outside: whether this point left the law's band, the consumer's own judgement.
    reading: the measurements behind the misfit, carried onto this point's receipt row.
    """

    loss: float
    outside: bool = False
    reading: dict[str, JsonValue] = {}


class Suggests(Protocol):
    """The one method a proposer calls on a driver's trial, typed here as drivers are optional."""

    def suggest_categorical(self, name: str, choices: Sequence[JsonValue]) -> JsonValue: ...


class Proposer(Protocol):
    """Whatever proposes a search's next point and is told what that point scored."""

    def ask(self) -> dict[str, JsonValue]:
        """The next point to evaluate, one value per axis of the declared space."""

    def tell(self, point: Mapping[str, JsonValue], loss: float) -> None:
        """What that point scored, which is the whole of what steers the next ask."""


class Optuna:
    """optuna as a proposer over a categorical space, its own storage left unopened.

    space: one axis per name, each the values it admits. Categorical because a search lane walks
        SHAPES, and a continuous axis rounded onto a legal shape lies to the sampler.
    seed: the sampler's seed, receipted by the study so the same walk can be taken again.
    """

    def __init__(self, space: Mapping[str, Sequence[JsonValue]], *, seed: int) -> None:
        optuna = driver("search")
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        self.space = {name: list(values) for name, values in space.items()}
        self.seed = seed
        self.study = optuna.create_study(
            direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed)
        )
        self.pending: Suggests | None = None

    def ask(self) -> dict[str, JsonValue]:
        self.pending = suggested = self.study.ask()
        return {
            name: suggested.suggest_categorical(name, values)
            for name, values in self.space.items()
        }

    def tell(self, point: Mapping[str, JsonValue], loss: float) -> None:
        """Tell the sampler what its last point scored, then release that point.

        A driver's trial object keeps the suggesting frame, and so a test's fixtures, reachable.
        """
        told, self.pending = self.pending, None
        self.study.tell(told, loss)


class Study:
    """One search lane: a budgeted worst-case hunt whose every iteration is a receipt row.

    question: the search in one sentence, which the receipt states was hunted.
    budget: how many evaluations the lane may spend.
    seed: the number that replays this search.
    refuted: the consumer's word for a worst point that left the band.
    survived: the consumer's word for a study whose points all stayed inside, which must not be a
        validation word; every point row also rides it, so a study narrates as it goes.
    owed: the declared cell that owes the worst point its confirmation, on fresh seeds.
    context: fields every row carries beside the search.
    """

    def __init__(
        self,
        trial: Trial,
        proposer: Proposer,
        *,
        question: str,
        budget: int,
        seed: int,
        refuted: str,
        survived: str,
        owed: Owed,
        **context: JsonValue,
    ) -> None:
        self.trial = trial
        self.proposer = proposer
        self.question = question
        self.budget = budget
        self.seed = seed
        self.refuted = refuted
        self.survived = survived
        self.owed = owed
        self.context = context
        self.points: list[tuple[dict[str, JsonValue], Miss]] = []

    @property
    def worst(self) -> tuple[dict[str, JsonValue], Miss]:
        """The point that scored the highest misfit."""
        return max(self.points, key=lambda taken: taken[1].loss)

    def run(self, evaluate: Callable[..., Miss]) -> tuple[dict[str, JsonValue], Miss]:
        """Spend the budget, write a row per point, settle the study, and return the worst.

        evaluate: takes one point by keyword and returns what it scored; the whole of the science.
        """
        for index in range(self.budget):
            point = self.proposer.ask()
            miss = evaluate(**point)
            self.proposer.tell(point, miss.loss)
            self.points.append((point, miss))
            self.narrate(index, point, miss)
        return self.settle()

    def narrate(self, index: int, point: Mapping[str, JsonValue], miss: Miss) -> None:
        """Write one ask-tell iteration's row, the record the sampler does not keep."""
        getattr(self.trial, self.survived)(
            f"point {index + 1} of {self.budget} at {dict(point)} scored a misfit of "
            f"{miss.loss:.6g} and landed {'OUTSIDE' if miss.outside else 'inside'} the band",
            **self.receipt(),
            row="point",
            point=dict(point),
            loss=miss.loss,
            outside=miss.outside,
            index=index + 1,
            **miss.reading,
        )

    def owes(self, point: Mapping[str, JsonValue]) -> Owed:
        """The confirmation debt with the worst point folded into its cell, so none is retyped."""
        drawn = {name: str(value) for name, value in point.items()}
        return self.owed.model_copy(update={"cell": {**self.owed.cell, **drawn}})

    def settle(self) -> tuple[dict[str, JsonValue], Miss]:
        """Settle the study on its worst point, refuted where that point left the band."""
        point, miss = self.worst
        escaped = [taken for taken, scored in self.points if scored.outside]
        owed = self.owes(point)
        word = self.refuted if miss.outside else self.survived
        verdict = (
            f"the worst of {len(self.points)} points is {dict(point)} at a misfit of "
            f"{miss.loss:.6g}, "
            + (
                f"which is OUTSIDE the band, as are {len(escaped)} of the points visited. It is "
                f"{owed.stated}"
                if miss.outside
                else "which is inside the band, so this search found no excursion inside its "
                "budget and that is a statement about the search rather than about the law"
            )
        )
        getattr(self.trial, word)(
            f"{self.question}: {verdict}",
            **self.receipt(),
            row="study",
            point=dict(point),
            loss=miss.loss,
            outside=miss.outside,
            index=len(self.points),
            outside_points=len(escaped),
            owed=owed.model_dump() if miss.outside else None,
            **miss.reading,
        )
        return point, miss

    def receipt(self) -> dict[str, JsonValue]:
        """The fields every row of this study carries, which make it replayable."""
        return {
            "lane_kind": "search",
            "question": self.question,
            "budget": self.budget,
            "seed": self.seed,
            "driver": "optuna",
            "replay": f"re-run this lane at seed {self.seed} with budget={self.budget}",
            **self.context,
        }
