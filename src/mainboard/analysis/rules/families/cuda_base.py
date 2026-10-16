"""What every stored relation of the PY-CUDA facts shares: a project that imports no GPU framework
has no row in any of them."""

import re

from ...catalog import Relation

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


class CudaRelation(Relation):
    """A relation of the PY-CUDA facts; its rows exist where some file imports a GPU framework
    (a stored one is built empty elsewhere)."""

    evidence_imports = GPU_MODULES
