from contextlib import suppress
from typing import TYPE_CHECKING

from ...facts.memory.core import Memory
from . import apis

if TYPE_CHECKING:
    from .gpu import NvidiaGPU


class Vram:
    """The device memory of one NVIDIA GPU, read through the first layer that answers."""

    def __init__(self, gpu: NvidiaGPU) -> None:
        self.gpu = gpu

    def _reading(self, source: str, total: int, *, used: int, free: int) -> Memory:
        """A device memory reading from `source`, flagged unified on a coherent pool."""
        return Memory(
            scope="vram",
            total_bytes=total,
            used_bytes=used,
            free_bytes=free,
            unified=self.gpu.coherent,
            source=source,
        )

    def nvml(self) -> Memory:
        """Current memory state from NVML when `cuda.core` is unavailable."""
        with suppress(*apis.nvml_errors):
            memory = apis.nvml.device_get_memory_info_v2(self.gpu.handle)
            return self._reading("nvml", memory.total, used=memory.used, free=memory.free)
        return self.runtime()

    def read(self) -> Memory:
        """CUDA-visible GPU memory: `cuda.core.system`, then NVML, then `cudaMemGetInfo`.

        On GH200 and other coherent platforms this counts HBM-resident allocations (the
        discrete-device counter) and carries `unified=True`, the probed signal that host RAM is
        a peer pool. Managed memory paged into Grace LPDDR is not counted, matching `nvidia-smi`.
        """
        if not apis.has_cuda_core:
            return self.nvml()
        try:
            memory = self.gpu.system_device.memory_info
        except apis.system.NotSupportedError:
            return self.runtime()
        return self._reading("cuda-core-system", memory.total, used=memory.used, free=memory.free)

    def runtime(self) -> Memory:
        """Current memory state from CUDA Runtime, the last resort when NVML memory refuses."""
        if not apis.has_runtime:
            raise RuntimeError("CUDA Runtime is unavailable and NVML refused its memory reading")
        runtime, index = apis.runtime, self.gpu.index
        err, current = runtime.cudaGetDevice()
        if err != runtime.cudaError_t.cudaSuccess:
            current = index
        runtime.cudaSetDevice(index)
        try:
            err, free_bytes, total_bytes = runtime.cudaMemGetInfo()
        finally:
            runtime.cudaSetDevice(current)
        if err != runtime.cudaError_t.cudaSuccess:
            raise RuntimeError(f"cudaMemGetInfo({index}) failed: {err}")
        return self._reading(
            "cuda-runtime", total_bytes, used=total_bytes - free_bytes, free=free_bytes
        )
