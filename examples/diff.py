"""Compare two runs to see what got faster. Runs anywhere."""

import time

from mainboard.profiling import Profiler, region


def run(compute_ms: int):
    with Profiler() as p:
        with region("load"):
            time.sleep(0.05)
        with region("compute"):
            time.sleep(compute_ms / 1000)
    return p.result()


baseline = run(100)
optimized = run(40)
optimized.diff(baseline).show()  # green where faster, red where slower
