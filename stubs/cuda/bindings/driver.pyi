# Stub for `cuda.bindings.driver` (cuda-bindings 13.x, no types shipped): the one driver call the
# CUPTI collector makes, to learn which context the calling thread has bound.

import enum

class CUresult(enum.IntEnum):
    CUDA_SUCCESS = ...

class CUcontext:
    def __int__(self) -> int: ...

def cuCtxGetCurrent() -> tuple[CUresult, CUcontext]: ...
