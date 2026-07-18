"""Time named regions and print where the time goes. Runs anywhere."""

import time

from mainboard.profiling import Profiler, span

with Profiler() as p:
    with span("load"):
        time.sleep(0.05)
    with span("compute"):
        time.sleep(0.10)

p.show()
