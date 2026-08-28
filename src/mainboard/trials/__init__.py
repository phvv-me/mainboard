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

from .coverage import PROBED, Cell, LaneStatus, Probed
from .dataset import Dataset
from .declaration import MARKERS, Declaration
from .distribution import Distribution, Fleet, Local, Partition
from .figures import Figures, Gap, Need, Refusal, rendered_twice
from .flags import Flag, held, moved, reading
from .ledger import NESTED, Ledger, TrialReceipts, wire
from .provenance import SOURCE_VAR, Card, Source, card_of, installed, provenance, source
from .session import Session, Trial
from .stage import Stage
from .universe import Universe
from .vocabulary import Outcome, Stance, Vocabulary, Word

__all__ = [
    "MARKERS",
    "NESTED",
    "PROBED",
    "SOURCE_VAR",
    "Card",
    "Cell",
    "Dataset",
    "Declaration",
    "Distribution",
    "Figures",
    "Flag",
    "Fleet",
    "Gap",
    "LaneStatus",
    "Ledger",
    "Local",
    "Need",
    "Outcome",
    "Partition",
    "Probed",
    "Refusal",
    "Session",
    "Source",
    "Stage",
    "Stance",
    "Trial",
    "TrialReceipts",
    "Universe",
    "Vocabulary",
    "Word",
    "card_of",
    "held",
    "installed",
    "moved",
    "provenance",
    "reading",
    "rendered_twice",
    "source",
    "wire",
]
