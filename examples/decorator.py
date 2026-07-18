"""Leave dormant spans in functions and activate them for one bounded profile."""

import time

from mainboard.profiling import Profiler, span


@span
def load() -> None:
    time.sleep(0.05)


@span
def compute() -> None:
    time.sleep(0.10)


with Profiler() as p:
    load()
    compute()
    compute()

p.show()  # `compute` shows calls=2 with its total and average
