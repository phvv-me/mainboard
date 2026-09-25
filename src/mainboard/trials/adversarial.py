# Adversarial lanes: a law stated as a property, and a draw budget spent trying to break it.
#
# The cells are the ones a SHRINKER chooses: draws are biased toward the operands' edges, and a
# failure is minimised to the smallest operand that still breaks the law. That witness is the
# artifact, because "it failed somewhere in eleven million elements" is a rumour and a witness a
# reader can retype is a finding.
#
# One lane is one trial and settles once, with draws, seed, budget and witness on the receipt. A
# find settles the consumer's refuted word with the witness; a survival settles its survival word
# with the draw count, never a validation word, since surviving two hundred draws is a statement
# about a search, not a population. The find is a candidate under the rule `adaptive` states.
# The seed is receipted and the example database is off, so re-running at that seed walks the
# same draws in the same order.

from typing import TYPE_CHECKING

from .adaptive import Owed, driver

if TYPE_CHECKING:
    from collections.abc import Callable

    from pydantic import JsonValue

    from .session import Trial


class Breach(Exception):
    """A law's own refusal, carrying the operand that broke it as receipt fields.

    Raised rather than returned, because a shrinker minimises on an exception. The witness is the
    draw's JSON description, never the draw itself, since a tensor is not a receipt column.

    witness: the operand as receipt fields.
    """

    def __init__(self, reason: str, **witness: JsonValue) -> None:
        super().__init__(reason)
        self.reason = reason
        self.witness = dict(witness)


class Hunt:
    """One adversarial lane: a law hunted for a counterexample under a stated draw budget.

    law: the property in one sentence, which the receipt states was hunted.
    draws: how many operands the hunt may draw, the whole of its budget.
    seed: the number that replays this search.
    refuted: the consumer's word for a law that broke.
    survived: the consumer's word for a law that held, which must not be a validation word.
    owed: the declared cell that owes a find its confirmation, on fresh seeds.
    context: fields both receipts carry beside the hunt.
    """

    def __init__(
        self,
        trial: Trial,
        *,
        law: str,
        draws: int,
        seed: int,
        refuted: str,
        survived: str,
        owed: Owed,
        **context: JsonValue,
    ) -> None:
        self.trial = trial
        self.law = law
        self.draws = draws
        self.seed = seed
        self.refuted = refuted
        self.survived = survived
        self.owed = owed
        self.context = context
        self.calls = 0
        self.until = 0

    def against(self, law: Callable[..., None], **operands: object) -> Breach | None:
        """Hunt `law` over `operands`, settle this trial, and return the witness or None.

        law: the property, taking the drawn operands by name and raising `Breach` when it fails.
        operands: one strategy per name the property takes.

        Every health check is suppressed: a draw against real silicon trips the slowness guard on
        a hunt working as intended, and a lane that filters hard is filtering toward its edge.
        """
        hypothesis = driver("adversarial")
        found: list[Breach] = []

        def probe(**drawn: object) -> None:
            self.calls += 1
            try:
                law(**drawn)
            except Breach as breach:
                self.until = self.until or self.calls
                found.append(breach)
                raise

        bounded = hypothesis.settings(
            max_examples=self.draws,
            database=None,
            deadline=None,
            report_multiple_bugs=False,
            suppress_health_check=list(hypothesis.HealthCheck),
        )
        hunted = hypothesis.seed(self.seed)(bounded(hypothesis.given(**operands)(probe)))
        try:
            hunted()
        except Breach:
            return self.broke(found[-1])
        self.held()
        return None

    def broke(self, witness: Breach) -> Breach:
        """Settle a find: the refuted word, the minimal witness, and the confirmation it owes.

        Settled through the word as a method, so an undeclared word is refused with the whole
        declared table printed.
        """
        getattr(self.trial, self.refuted)(
            f"{self.law} BROKE at draw {self.until} of {self.draws}, and shrank over "
            f"{self.calls - self.until} further calls to {witness.reason}. It is "
            f"{self.owed.stated}",
            **self.receipt(),
            witness=witness.witness,
            broke_at=self.until,
            shrinks=self.calls - self.until,
            owed=self.owed.model_dump(),
        )
        return witness

    def held(self) -> None:
        """Settle a survival: the survival word, the draws taken, and no claim beyond them."""
        getattr(self.trial, self.survived)(
            f"{self.law} survived {self.calls} of {self.draws} budgeted edge-biased draws at "
            f"seed {self.seed}, which is a statement about this search and not about a "
            f"population, so nothing here is coverage",
            **self.receipt(),
            witness={},
            broke_at=None,
            shrinks=0,
            owed=None,
        )

    def receipt(self) -> dict[str, JsonValue]:
        """The fields both outcomes carry, which make this hunt replayable."""
        return {
            "lane_kind": "adversarial",
            "law": self.law,
            "draws": self.draws,
            "draws_taken": self.calls,
            "seed": self.seed,
            "driver": "hypothesis",
            "replay": f"re-run this lane at seed {self.seed} with draws={self.draws}",
            **self.context,
        }
