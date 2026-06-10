from __future__ import annotations

import platform
from functools import cache, cached_property

import psutil
from pydantic import Field

from ...enums import Vendor
from ...gpu import GPU
from ...models.clock import Clock
from ...models.memory import Memory
from ...shell import ProfileRecord
from . import profile


class AppleGPU(GPU):
    """Apple Silicon integrated GPU backed by unified memory."""

    vendor: Vendor = Field(default=Vendor.APPLE)
    backend: str = "metal"

    @classmethod
    def is_available(cls) -> bool:
        """Whether this host reports an Apple Silicon GPU."""
        if platform.system() != "Darwin" or platform.machine() != "arm64":
            return False
        return bool(cls.gpu_records())

    @classmethod
    @cache
    def gpu_records(cls) -> tuple[ProfileRecord, ...]:
        """Apple GPU records from `system_profiler`."""
        if platform.system() != "Darwin":
            return ()
        records = profile.apple_system_profile().get("SPDisplaysDataType", [])
        return tuple(
            record for record in records if record.get("sppci_device_type") == "spdisplays_gpu"
        )

    @classmethod
    def all(cls) -> tuple[AppleGPU, ...]:
        """Return Apple Silicon GPUs reported by macOS."""
        return tuple(cls(index=i) for i, _ in enumerate(cls.gpu_records()))

    @cached_property
    def record(self) -> ProfileRecord:
        """Raw `system_profiler` display record."""
        return self.gpu_records()[self.index]

    @cached_property
    def name(self) -> str:
        """Apple GPU model name."""
        return str(self.record.get("sppci_model") or self.record.get("_name") or "Apple GPU")

    @cached_property
    def uuid(self) -> str:
        """Stable system UUID used as the integrated GPU identifier."""
        hardware = profile.apple_system_profile().get("SPHardwareDataType", [{}])[0]
        return str(hardware.get("platform_UUID") or "")

    @cached_property
    def architecture(self) -> str:
        """Apple SoC family backing this GPU."""
        hardware = profile.apple_system_profile().get("SPHardwareDataType", [{}])[0]
        return str(hardware.get("chip_type") or self.name)

    @cached_property
    def core_count(self) -> int:
        """Number of Apple GPU cores (the profiler reports it as a numeric string)."""
        try:
            return int(str(self.record.get("sppci_cores") or 0))
        except ValueError:
            return 0

    @cached_property
    def metal_support(self) -> str:
        """Metal support string reported by macOS."""
        return str(self.record.get("spdisplays_mtlgpufamilysupport") or "")

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
        """Apple GPU clocks are not exposed without privileged sampling."""
        return (
            Clock(domain="gpu_compute", source="system_profiler", supported=False),
            Clock(domain="memory", source="system_profiler", supported=False),
        )
