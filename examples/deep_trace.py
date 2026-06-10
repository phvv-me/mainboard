"""Deep per-kernel trace on CUDA, exported to a Perfetto timeline.

Needs a CUDA GPU + torch. ``trace=True`` adds the CUPTI Activity collector (off by
default). On macOS/AMD the regions and snapshots still work; the per-kernel tier is
empty until that backend lands (see the roadmap).
"""

import torch

from mainboard.profiling import Profiler, region

x = torch.randn(4096, 4096, device="cuda")

with Profiler(trace=True) as p, region("matmul"):
    for _ in range(10):
        y = x @ x
    torch.cuda.synchronize()

result = p.result()
result.show()  # regions + hot kernels
result.perfetto("trace.json")  # open at https://ui.perfetto.dev
