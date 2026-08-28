# THE TRIALS SUBSYSTEM: what a measured trial writes down, and how anyone reads it back.
#
# This is compute infrastructure and not anyone's science. A receipt store, a completeness rule, a
# settle vocabulary and a render contract are the same machinery whether the trials behind them
# are quantization sweeps, numeric reproducibility claims or kernel benchmarks, so they live with
# the tool that already owns the `trial_receipt` wire contract, the `verdict` verb and tracking.
#
# NOTHING HERE NAMES A CONSUMER. The settle words are configuration, the coverage axes are
# configuration, the tracked flags are configuration and the storage layout is configuration. What
# is fixed is the two outcomes, `passed` and `failed`, because those are what an exit code is
# derived from, and the shape of the printed receipt line, because that is a contract other tools
# already read. A project joins by declaring, never by being known about here.
#
# A consumer's conftest is one registration line and one hook. The whole of it:
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
# A lane is an ordinary test function that asks for `trial` and supplies measurements, and the
# declared words are methods on it. Nothing in a lane names a run, a card, a commit, a claim or a
# tracked flag, because every one of those is derived.
#
# TWO LANE KINDS CHOOSE THEIR OWN CELLS AND ARE OPTIONAL EXTRAS. `Hunt` states a law as a property
# and spends a draw budget trying to break it, shrinking a failure to a minimal witness; `Study`
# spends a budget of real evaluations walking toward a worst case and writes a receipt row per
# ask-tell iteration. Their drivers, hypothesis and optuna, are extras this tool does not depend
# on, and a lane declared without one refuses by naming the package and the extra. Both obey the
# rule `adaptive` states: an adaptive result is a CANDIDATE, never coverage, and it is confirmed by
# a declared parametrize cell on fresh seeds before any claim leans on it.

from .adaptive import Absent, Owed, driver
from .adversarial import Breach, Hunt
from .coverage import PROBED, Cell, LaneStatus, Probed
from .dataset import Dataset
from .declaration import MARKERS, Declaration
from .distribution import Distribution, Fleet, Local, Partition
from .figures import Figures, Gap, Need, Refusal, rendered_twice
from .flags import Flag, held, moved, reading
from .ledger import NESTED, Ledger, TrialReceipts, wire
from .lints import Finding, findings
from .provenance import SOURCE_VAR, Card, Source, card_of, installed, provenance, source
from .search import Miss, Optuna, Proposer, Study
from .session import Session, Trial
from .stage import Stage
from .universe import Universe
from .vocabulary import Outcome, Stance, Vocabulary, Word

__all__ = [
    "MARKERS",
    "NESTED",
    "PROBED",
    "SOURCE_VAR",
    "Absent",
    "Breach",
    "Card",
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
    "driver",
    "findings",
    "held",
    "installed",
    "moved",
    "provenance",
    "reading",
    "rendered_twice",
    "source",
    "wire",
]
