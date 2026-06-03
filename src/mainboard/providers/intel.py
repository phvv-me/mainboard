from __future__ import annotations

from pydantic import Field

from ..enums import Vendor
from ..gpu import GPU
from ..npu import NPU


class IntelGPU(GPU):
    """Intel GPU provider stub for future Level Zero / oneAPI support."""

    vendor: Vendor = Field(default=Vendor.INTEL)
    backend: str = "level-zero"

    @classmethod
    def is_available(cls) -> bool:
        """Whether Intel GPUs are available through the future provider."""
        return False

    @classmethod
    def all(cls) -> tuple[IntelGPU, ...]:
        """Return detected Intel GPUs; empty until the provider is implemented."""
        return ()


class IntelNPU(NPU):
    """Intel NPU provider stub for future OpenVINO / Level Zero support."""

    vendor: Vendor = Field(default=Vendor.INTEL)
    backend: str = "openvino"

    @classmethod
    def is_available(cls) -> bool:
        """Whether Intel NPUs are available through the future provider."""
        return False

    @classmethod
    def all(cls) -> tuple[IntelNPU, ...]:
        """Return detected Intel NPUs; empty until the provider is implemented."""
        return ()
