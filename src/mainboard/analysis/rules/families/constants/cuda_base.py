"""The constants of `cuda_base` other modules share."""

import re

GPU_MODULES = frozenset(
    {
        "cupy",
        "cupyx",
        "torch",
        "numba",
        "cuda",
        "triton",
        "helion",
        "warp",
        "jax",
        "tilelang",
        "patos.cuda",
        "patos.torch",
    }
)

GPU_MODULE_PATTERN = "^(" + "|".join(sorted(map(re.escape, GPU_MODULES))) + ")(\\.|$)"

NUMBA_CUDA = r"(numba\.cuda|patos\.cuda\.typed\.cuda)"
