from __future__ import annotations

import platform
from functools import cached_property

import psutil
from pydantic import Field

from ...enums import Vendor
from ...models.clock import Clock
from ...models.memory import Memory
from ...npu import NPU
from . import profile


class AppleNPU(NPU):
    """Apple Neural Engine backed by unified memory."""

    vendor: Vendor = Field(default=Vendor.APPLE)
    backend: str = "coreml"

    @classmethod
    def is_available(cls) -> bool:
        """Whether this host is an Apple Silicon machine."""
        return platform.system() == "Darwin" and platform.machine() == "arm64"

    @classmethod
    def all(cls) -> tuple[AppleNPU, ...]:
        """Return the local Apple Neural Engine when present."""
        return (cls(),) if cls.is_available() else ()

    @cached_property
    def name(self) -> str:
        """Apple Neural Engine model name."""
        hardware = profile.apple_system_profile().get("SPHardwareDataType", [{}])[0]
        chip = hardware.get("chip_type") or "Apple Silicon"
        return f"{chip} Neural Engine"

    @cached_property
    def architecture(self) -> str:
        """Apple SoC family backing the Neural Engine."""
        hardware = profile.apple_system_profile().get("SPHardwareDataType", [{}])[0]
        return str(hardware.get("chip_type") or "Apple Silicon")

    @property
    def memory(self) -> Memory:
        """Unified memory visible to CPU, GPU, and Neural Engine."""
        vm = psutil.virtual_memory()
        return Memory(
            scope="unified",
            total_bytes=vm.total,
            used_bytes=vm.used,
            free_bytes=vm.available,
            unified=True,
            source="psutil",
        )

    @property
    def clock_readings(self) -> tuple[Clock, ...]:
        """Apple Neural Engine clocks are not exposed through public APIs."""
        return (Clock(domain="npu", source="coreml", supported=False),)
