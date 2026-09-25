# The vocabulary a study author declares an experiment with: config-domain markers, the four
# gates, a counterbalanced lane, the per-trial context, and a decorated function or hand-written
# `Experiment` subclass. Driver machinery (`GateVerdict`, `space_of`, `orders`/`validates`,
# `runnable`, the trial outcomes) stays in its own submodule, off this headline surface. What a
# trial tells the outside world is its one `board_surface.RECEIPT` line.

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
