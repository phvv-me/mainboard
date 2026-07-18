"""Compare two runs to see what got faster. Runs anywhere."""

import time

from mainboard.profiling import Profile, Profiler, span


def run(compute_ms: int) -> Profile:
    with Profiler() as p:
        with span("load"):
            time.sleep(0.05)
        with span("compute"):
            time.sleep(compute_ms / 1000)
    return p.result()


baseline = run(100)
optimized = run(40)
optimized.diff(baseline).show()  # green where faster, red where slower
