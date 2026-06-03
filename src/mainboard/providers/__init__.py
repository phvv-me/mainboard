from __future__ import annotations

from ..gpu import GPU
from ..npu import NPU
from .amd import AMDGPU
from .apple import AppleGPU, AppleNPU
from .intel import IntelGPU, IntelNPU
from .nvidia import NvidiaGPU
from .qualcomm import QualcommGPU, QualcommNPU

GPU.register_providers(NvidiaGPU, AppleGPU, AMDGPU, IntelGPU, QualcommGPU)
NPU.register_providers(AppleNPU, IntelNPU, QualcommNPU)

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
