"""Time named regions and print where the time goes. Runs anywhere."""

import time

from mainboard.profiling import Profiler, region

with Profiler() as p:
    with region("load"):
        time.sleep(0.05)
    with region("compute"):
        time.sleep(0.10)

p.show()
