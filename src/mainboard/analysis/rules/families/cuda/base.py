"""What every stored relation of the PY-CUDA facts shares: a project that imports no GPU framework
has no row in any of them."""

from ....catalog import Relation
from ..constants.cuda_base import GPU_MODULES


class CudaRelation(Relation):
    """A relation of the PY-CUDA facts; its rows exist where some file imports a GPU framework
    (a stored one is built empty elsewhere)."""

    evidence_imports = GPU_MODULES
