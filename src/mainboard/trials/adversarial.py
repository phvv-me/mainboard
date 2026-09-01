# ADVERSARIAL LANES: A LAW STATED AS A PROPERTY, AND A BUDGET SPENT TRYING TO BREAK IT.
#
# A declared lane asks a law at the cells somebody chose. An adversarial lane asks it at the cells
# a SHRINKER chooses, which is the whole difference: the draws are biased toward the edges of
# whatever the operands are, and the failure, if one comes, is minimised until what is left is the
# smallest operand that still breaks the law. That minimal operand is the artifact. A refutation
# that arrives as "it failed somewhere in eleven million elements" is a rumour, and one that
# arrives as a witness a reader can retype is a finding.
#
# ONE LANE IS ONE TRIAL AND IT SETTLES ONCE. The draws, the seed, the budget and the witness ride
# on that one receipt, so a reader learns how hard the law was pushed and not merely that it was.
# A find settles the consumer's REFUTED word with the witness. A survival settles the consumer's
# SURVIVAL word with the draw count stated, and it is never a validation word: surviving two
# hundred adversarial draws is a statement about a search, not about a population, and spelling it
# `validated` would let a hunt masquerade as coverage.
#
# THE BINDING DISCIPLINE, WHICH `adaptive` STATES IN FULL AND THIS MODULE OBEYS. An adaptive result
# is a CANDIDATE, never coverage. A witness this lane minimises is confirmed by a declared
# parametrize cell on FRESH SEEDS before any claim leans on it, and `Owed` is where that debt is
# written down. Search proposes and the grid confirms.
#
# AND IT REPLAYS FROM ITS OWN RECEIPT. The seed is receipted and the example database is turned
# off, so the search is a pure function of the seed, the budget and the strategies: re-running the
# lane at the receipted seed walks the same draws in the same order. A hunt whose receipt cannot
# replay the hunt is an anecdote.
#
# HYPOTHESIS IS AN OPTIONAL EXTRA and is reached through `adaptive.driver`, so a workspace that
# declares no adversarial lane installs no shrinker and one that declares a lane without the
# package gets a refusal naming both.

from typing import TYPE_CHECKING

from .adaptive import Owed, driver

if TYPE_CHECKING:
    from collections.abc import Callable

    from pydantic import JsonValue

    from .session import Trial


class Breach(Exception):
    """A law's own refusal, carrying the drawn operand that broke it as receipt fields.

    The law raises this instead of returning a verdict, because a shrinker minimises on an
    exception and there is no second channel it watches. What rides on it is the WITNESS as JSON,
    the description of the draw and never the draw itself, since a tensor is not a receipt column
    and a witness a reader cannot retype is not an artifact.

    reason: one line saying what the law found. witness: the operand as receipt fields.
    """

    def __init__(self, reason: str, **witness: JsonValue) -> None:
        super().__init__(reason)
        self.reason = reason
        self.witness = dict(witness)


class Hunt:
    """One adversarial lane: a law hunted for a counterexample under a stated draw budget.

    trial: the evidence line this lane settles.
    law: the property in one sentence, which is what the receipt states was hunted.
    draws: how many operands the hunt may draw, which is the whole of its budget.
    seed: the number that replays this search, receipted so it can.
    refuted: the consumer's word for a law that broke.
    survived: the consumer's word for a law that held, which must not be a validation word.
    owed: the declared cell that owes a find its confirmation, on fresh seeds.
    context: fields both receipts carry, whatever the lane wants a reader to see beside the hunt.
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
        """Hunt `law` over `operands`, settle this trial, and hand back the witness or nothing.

        law: the property, taking the drawn operands by name and raising `Breach` when it fails.
        operands: one strategy per name the property takes, whose edge bias is the lane's own.

        Every health check is suppressed on purpose: an operand drawn against real silicon takes
        long enough that the slowness guard fires on a hunt that is working exactly as intended,
        and a lane that filters hard is filtering toward the edge it was written to probe.
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

        Settled through the word as a method rather than through `settle`, so a lane naming a
        word its workspace never declared is refused with the whole declared table printed.
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
        """The fields both outcomes carry, which is what makes this hunt replayable."""
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
