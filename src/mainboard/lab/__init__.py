# The declaring vocabulary for an experiment: the config-domain markers, the four unified
# gates, a counterbalanced lane, the per-trial context, and the two ways to declare an
# `Experiment` (a decorated function or a hand-written subclass). Supporting machinery
# (`GateVerdict`, `space_of`, `orders`/`validates`, `runnable`, the trial outcome types) stays
# reachable through its own submodule for a driver that needs it, kept out of this headline
# surface so the vocabulary a study author writes against stays exactly this list.

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
