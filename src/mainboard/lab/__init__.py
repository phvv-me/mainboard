# The declaring vocabulary for an experiment: the config-domain markers, the four unified
# gates, a counterbalanced lane, the per-trial context, and the two ways to declare an
# `Experiment` (a decorated function or a hand-written subclass). Supporting machinery
# (`GateVerdict`, `space_of`, `orders`/`validates`, `runnable`, the trial outcome types) stays
# reachable through its own submodule for a driver that needs it, kept out of this headline
# surface so the vocabulary a study author writes against stays exactly this list.
#
# What a trial says to the outside world is one line. Every outcome renders itself through
# `TrialOutcome.receipt()` as a single JSON object under `board_surface.RECEIPT`, carrying the
# trial's content-addressed `run_id`, its outcome word, every declared gate's verdict, and
# whatever the outcome kind itself holds. A driver prints it; nothing here does. That line is
# the whole contract with anything watching a study's output, which is how atpx turns a
# verified claim's run into evidence naming the trial behind it without importing this package.

from .board_surface import experiment
from .domains import Choices, Fixed, FloatRange, IntRange
from .experiment import Experiment
from .gates import Idle, Offline, Parity, Receipt
from .lane import Lane
from .run import Run

__all__ = [
    "Choices",
    "Experiment",
    "Fixed",
    "FloatRange",
    "Idle",
    "IntRange",
    "Lane",
    "Offline",
    "Parity",
    "Receipt",
    "Run",
    "experiment",
]
