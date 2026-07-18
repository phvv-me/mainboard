"""Deep per-kernel trace on CUDA, exported to a Perfetto timeline.

Needs a CUDA GPU and torch. `Profiler.Feature.ACTIVITY` adds the CUPTI Activity
collector. Other feature flags independently control timing, device telemetry, and
native markers.
"""

import torch

from mainboard.profiling import Profiler, span

x = torch.randn(4096, 4096, device="cuda")

with (
    Profiler(
        features=(
            Profiler.Feature.SPANS
            | Profiler.Feature.DEVICE
            | Profiler.Feature.MARKERS
            | Profiler.Feature.ACTIVITY
        ),
        activities=Profiler.Activity.DEFAULT,
    ) as p,
    span("matmul"),
):
    for _ in range(10):
        y = x @ x
    torch.cuda.synchronize()

result = p.result()
result.show()  # regions + hot kernels
result.perfetto("trace.json")  # open at https://ui.perfetto.dev
