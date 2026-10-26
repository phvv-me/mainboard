# The trials subsystem: what a measured trial writes down, and how anyone reads it back.
#
# Compute infrastructure, not anyone's science: a receipt store, completeness rule, settle
# vocabulary and render contract live with the tool that owns the `trial_receipt` wire contract,
# the `verdict` verb and tracking. Nothing here names a consumer. Settle words, coverage axes,
# tracked flags and storage layout are configuration; fixed are only the outcomes `passed` and
# `failed` (an exit code derives from them) and the printed receipt line other tools read.
#
# A consumer's whole conftest:
#
#     from mainboard.trials import Declaration, Universe, Vocabulary
#
#     pytest_plugins = ["mainboard.trials.pytest_plugin"]
#
#     def pytest_trials_declaration() -> Declaration:
#         return Declaration(
#             universe=Universe(root=HERE, axes=("card", "model"), probed=("torch", "triton")),
#             words=Vocabulary.of("validated", "refuted", "known", "abandoned"),
#             flags=ARITHMETIC,
#         )
#
# A lane is a test function that asks for `trial` and supplies measurements; the declared words
# are methods on it, and run, card, commit, claim and tracked flags are all derived. A receipt
# identifies the captured source by SHA-256 and registered input rows by digest, so new and edited
# files qualify by their captured bytes, not their version-control state. Historical rejected
# receipts keep their original labels. The optional adaptive lanes, `Hunt` and `Study`, choose
# their own cells under the rule `adaptive` states.

from ..dispatch.constants import SOURCE_VAR
from .artifacts import Artifact
from .contract.declaration import Declaration
from .contract.figures import Figures, Gap, Need, Refusal, rendered_twice
from .contract.flags import Flag, held, moved, reading
from .contract.vocabulary import Outcome, Stance, Vocabulary, Word
from .lanes.adaptive import Absent, Owed, driver
from .lanes.adversarial import Breach, Hunt
from .lanes.coverage import Cell, LaneStatus, Probed
from .lanes.search import Miss, Optuna, Study
from .provenance import (
    Admissibility,
    Card,
    Preflight,
    Source,
    card_of,
    digest_of,
    digested,
    installed,
    source,
)
from .running.lease import Busy, CardLease
from .running.lints import Finding, findings
from .running.log import Log
from .running.stage import Stage
from .session import Session, Trial
from .store.constants import ADMISSIBILITY, BASELINES, MARKERS, NESTED, OPENED, PROBED
from .store.dataset import Ambiguous, Dataset
from .store.ledger import Ledger, TrialReceipts, wire
from .store.universe import Universe

__all__ = [
    "Artifact",
    "Log",
    "ADMISSIBILITY",
    "BASELINES",
    "MARKERS",
    "NESTED",
    "OPENED",
    "PROBED",
    "SOURCE_VAR",
    "Absent",
    "Admissibility",
    "Ambiguous",
    "Breach",
    "Busy",
    "Card",
    "CardLease",
    "Cell",
    "Dataset",
    "Declaration",
    "Figures",
    "Flag",
    "Finding",
    "Gap",
    "Hunt",
    "LaneStatus",
    "Ledger",
    "Miss",
    "Need",
    "Optuna",
    "Outcome",
    "Owed",
    "Preflight",
    "Probed",
    "Refusal",
    "Session",
    "Source",
    "Stage",
    "Stance",
    "Study",
    "Trial",
    "TrialReceipts",
    "Universe",
    "Vocabulary",
    "Word",
    "card_of",
    "digest_of",
    "digested",
    "driver",
    "findings",
    "held",
    "installed",
    "moved",
    "reading",
    "rendered_twice",
    "source",
    "wire",
]
