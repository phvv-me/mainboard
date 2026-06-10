"""Profile functions with the @profile decorator — each call is a region."""

import time

from mainboard.profiling import Profiler, profile


@profile
def load() -> None:
    time.sleep(0.05)


@profile
def compute() -> None:
    time.sleep(0.10)


with Profiler() as p:
    load()
    compute()
    compute()

p.show()  # `compute` shows calls=2 with its total and average
