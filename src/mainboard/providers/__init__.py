from .amd import AMDGPU
from .apple import AppleGPU, AppleNPU
from .intel import IntelGPU, IntelNPU
from .nvidia import NvidiaGPU
from .qualcomm import QualcommGPU, QualcommNPU

__all__ = [
    "AMDGPU",
    "AppleGPU",
    "AppleNPU",
    "IntelGPU",
    "IntelNPU",
    "NvidiaGPU",
    "QualcommGPU",
    "QualcommNPU",
]
