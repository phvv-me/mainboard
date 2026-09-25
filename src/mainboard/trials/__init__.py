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

from ..dispatch.shared import SOURCE_VAR
from .adaptive import Absent, Owed, driver
from .adversarial import Breach, Hunt
from .artifacts import Artifact
from .coverage import PROBED, Cell, LaneStatus, Probed
from .dataset import ADMISSIBILITY, OPENED, Ambiguous, Dataset
from .declaration import MARKERS, Declaration
from .distribution import Distribution, Fleet, Local, Partition
from .figures import Figures, Gap, Need, Refusal, rendered_twice
from .flags import Flag, held, moved, reading
from .lease import Busy, CardLease
from .ledger import NESTED, Ledger, TrialReceipts, wire
from .lints import Finding, findings
from .log import Log
from .provenance import (
    BASELINES,
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
from .search import Miss, Optuna, Proposer, Study
from .session import Session, Trial
from .stage import Stage
from .universe import Universe
from .vocabulary import Outcome, Stance, Vocabulary, Word

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
    "Distribution",
    "Figures",
    "Flag",
    "Finding",
    "Fleet",
    "Gap",
    "Hunt",
    "LaneStatus",
    "Ledger",
    "Local",
    "Miss",
    "Need",
    "Optuna",
    "Outcome",
    "Owed",
    "Partition",
    "Preflight",
    "Probed",
    "Proposer",
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
