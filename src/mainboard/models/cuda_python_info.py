from ..enums import CudaPythonVariant
from .base import FrozenModel


class CudaPythonInfo(FrozenModel):
    """Detected CUDA Python stack variant and CUPTI availability."""

    variant: CudaPythonVariant
    supports_cupti: bool
