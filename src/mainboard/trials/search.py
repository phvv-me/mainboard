# SEARCH LANES: A WORST CASE PROPOSED ADAPTIVELY, ONE RECEIPT ROW PER EVALUATION.
#
# Where an adversarial lane draws cheap operands and shrinks a failure, a search lane spends a REAL
# EVALUATION per point, a GPU measurement that takes seconds and returns a continuous misfit rather
# than a pass or a fail. A grid over that is unaffordable at any resolution worth having, and a
# sampler steered by the misfit is what finds the worst corner inside a budget somebody can pay.
#
# THE LEDGER IS THE STORAGE AND THE SAMPLER IS NOT. A sampler that owned the record would put the
# evidence of a run in a second place, under a schema nobody here declared, with no card, no
# commit, no tracked flag and no coverage cell on it. So the driver is asked for one thing, the
# next point, and is told one thing, what that point scored; everything else is a receipt row
# written through the same trial and the same store every declared lane writes through. Ask-tell
# is exactly that seam and it needs no storage at all, which is why one is never opened.
#
# ONE RECEIPT ROW PER ASK-TELL ITERATION, RIDING SAMPLES-PER-CELL. A universe already declares how
# many readings one cell owes and already accumulates them across runs, so a study writing a row
# per point is that rule used rather than a second store invented. The study's own outcome is one
# more row on top, marked as such, and it is the row a reader reaches for first.
#
# THE LANE SETTLES ON THE STUDY AND NOT ON A POINT. A worst point that left the law's band settles
# the consumer's REFUTED word naming that point; otherwise the study settles the consumer's
# SURVIVAL word with the trial count stated. The budget is spent either way, because the object
# being reported is the WORST point and a search stopped at the first excursion has not found it.
#
# THE BINDING DISCIPLINE, WHICH `adaptive` STATES IN FULL AND THIS MODULE OBEYS. An adaptive result
# is a CANDIDATE, never coverage. A sampler walks toward the corner it is rewarded for, so the
# worst point it returns is a proposal about WHERE the law is weak and never a rate at which it is,
# and a claim leaning on it is owed a declared parametrize cell on FRESH SEEDS. Search proposes and
# the grid confirms.
#
# THE LOSS IS ALWAYS MAXIMISED, because a search lane hunts a worst case and a lane that wanted the
# best of something would be an optimiser rather than an experiment. A consumer whose misfit reads
# the other way negates it, which is one character against a direction knob nobody would ever set
# to `minimize` here.
#
# OPTUNA IS AN OPTIONAL EXTRA and is reached through `adaptive.driver`, so a workspace that
# declares no search lane installs no sampler and one that declares a lane without the package
# gets a refusal naming both.

from typing import TYPE_CHECKING, Protocol

from patos import FrozenModel

# Imported at runtime rather than under `TYPE_CHECKING` because `Miss` is a pydantic model and
# pydantic resolves a field's annotation when the class is built, so a deferred name is a model
# that refuses to construct.
from pydantic import JsonValue

from .adaptive import Owed, driver

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from .session import Trial


class Miss(FrozenModel):
    """What one evaluated point of a search scored, and what was read to score it.

    loss: the continuous misfit this point produced, which is what the sampler is steered by and
        what the study maximises.
    outside: whether this point left the law's band, which is the consumer's own judgement and
        never this module's, since a band is a fact about a claim.
    reading: the measurements behind the misfit, carried onto this point's receipt row.
    """

    loss: float
    outside: bool = False
    reading: dict[str, JsonValue] = {}


class Suggests(Protocol):
    """The one thing a proposer asks of a driver's own trial object, spelled so nothing imports it.

    A driver is an optional extra, so its types cannot be named at type-check time, and a search
    lane's proposer needs exactly one method of a suggested point. This is that method.
    """

    def suggest_categorical(self, name: str, choices: Sequence[JsonValue]) -> JsonValue:
        """One axis's value, drawn from the values that axis admits."""


class Proposer(Protocol):
    """Whatever proposes a search's next point and is told what that point scored.

    Two methods, because two is what a study needs. A driver that also ranks, prunes, plots or
    persists is welcome to, and a study that had to know about any of it would be a study coupled
    to one library.
    """

    def ask(self) -> dict[str, JsonValue]:
        """The next point to evaluate, one value per axis of the declared space."""

    def tell(self, point: Mapping[str, JsonValue], loss: float) -> None:
        """What that point scored, which is the whole of what steers the next ask."""


class Optuna:
    """optuna as a proposer over a categorical space, its own storage left unopened.

    space: one axis per name, each a sequence of the values that axis admits. Categorical because
        a search lane here walks SHAPES, a card runs the ones it runs, and a continuous axis
        rounded onto a legal shape is a sampler being lied to about what it proposed.
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
        """One point off the sampler, every axis suggested from the values it declared."""
        self.pending = suggested = self.study.ask()
        return {
            name: suggested.suggest_categorical(name, values)
            for name, values in self.space.items()
        }

    def tell(self, point: Mapping[str, JsonValue], loss: float) -> None:
        """Hand the sampler back what its own last point scored, then let that point go.

        A proposer has no use for a suggestion it has already scored, and a driver's trial object
        is not a small thing to keep, so the pending point is released here rather than left on
        this object until the next `ask` happens to overwrite it.
        """
        told, self.pending = self.pending, None
        self.study.tell(told, loss)


class Study:
    """One search lane: a budgeted worst-case hunt whose every iteration is a receipt row.

    trial: the evidence line this lane settles.
    proposer: what proposes the next point and is told what it scored.
    question: the search in one sentence, which is what the receipt states was hunted.
    budget: how many evaluations the lane may spend, which is the whole of its cost.
    seed: the number that replays this search, receipted so it can.
    refuted: the consumer's word for a worst point that left the band.
    survived: the consumer's word for a study whose points all stayed inside it, which must not be
        a validation word, and which every point row also rides so a study narrates as it goes.
    owed: the declared cell that owes the worst point its confirmation, on fresh seeds.
    context: fields every row carries, whatever the lane wants a reader to see beside the search.
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
        """The point that scored the highest misfit, which is what this study went to find."""
        return max(self.points, key=lambda taken: taken[1].loss)

    def run(self, evaluate: Callable[..., Miss]) -> tuple[dict[str, JsonValue], Miss]:
        """Spend the budget, write a row per point, settle the study, and hand back the worst.

        evaluate: takes one point by keyword and returns what it scored. It is the whole of the
        science and this loop is the whole of the machinery, which is the split that lets a lane
        change what it measures without touching anything here.
        """
        for index in range(self.budget):
            point = self.proposer.ask()
            miss = evaluate(**point)
            self.proposer.tell(point, miss.loss)
            self.points.append((point, miss))
            self.narrate(index, point, miss)
        return self.settle()

    def narrate(self, index: int, point: Mapping[str, JsonValue], miss: Miss) -> None:
        """Write one ask-tell iteration's row, which is the record the sampler does not keep."""
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
        """The confirmation debt with the worst point folded into the cell it names.

        A search's owed cell IS the point it found, and the study already holds that point, so
        naming it here rather than making a lane retype coordinates it does not yet know is the
        same rule the rest of this subsystem runs on: a fact a caller has to retype is a fact a
        caller will eventually retype wrong.
        """
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
        """The fields every row of this study carries, which is what makes it replayable."""
        return {
            "lane_kind": "search",
            "question": self.question,
            "budget": self.budget,
            "seed": self.seed,
            "driver": "optuna",
            "replay": f"re-run this lane at seed {self.seed} with budget={self.budget}",
            **self.context,
        }
