from __future__ import annotations

import json
import platform
from functools import cache, cached_property
from typing import Any

import psutil
from plumbum import local
from pydantic import Field

from ..enums import Vendor
from ..gpu import GPU
from ..models.clock import Clock
from ..models.mem_info import MemInfo
from ..models.memory_usage import MemoryUsage
from ..npu import NPU


@cache
def apple_system_profile() -> dict[str, Any]:
    """Return cached macOS hardware and display profiler data."""
    profiler = local["system_profiler"]
    return json.loads(profiler["-json", "SPHardwareDataType", "SPDisplaysDataType"]())


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
    def gpu_records(cls) -> tuple[dict[str, Any], ...]:
        """Apple GPU records from `system_profiler`."""
        if platform.system() != "Darwin":
            return ()
        try:
            records = apple_system_profile().get("SPDisplaysDataType", [])
        except (OSError, json.JSONDecodeError):
            return ()
        return tuple(
            record for record in records if record.get("sppci_device_type") == "spdisplays_gpu"
        )

    @classmethod
    def all(cls) -> tuple[AppleGPU, ...]:
        """Return Apple Silicon GPUs reported by macOS."""
        return tuple(cls(index=i) for i, _ in enumerate(cls.gpu_records()))

    @cached_property
    def record(self) -> dict[str, Any]:
        """Raw `system_profiler` display record."""
        return self.gpu_records()[self.index]

    @cached_property
    def name(self) -> str:
        """Apple GPU model name."""
        return str(self.record.get("sppci_model") or self.record.get("_name") or "Apple GPU")

    @cached_property
    def uuid(self) -> str:
        """Stable system UUID used as the integrated GPU identifier."""
        hardware = apple_system_profile().get("SPHardwareDataType", [{}])[0]
        return str(hardware.get("platform_UUID") or "")

    @cached_property
    def architecture(self) -> str:
        """Apple SoC family backing this GPU."""
        hardware = apple_system_profile().get("SPHardwareDataType", [{}])[0]
        return str(hardware.get("chip_type") or self.name)

    @cached_property
    def core_count(self) -> int:
        """Number of Apple GPU cores."""
        raw = self.record.get("sppci_cores") or 0
        try:
            return int(raw)
        except (TypeError, ValueError):
            return 0

    @cached_property
    def metal_support(self) -> str:
        """Metal support string reported by macOS."""
        return str(self.record.get("spdisplays_mtlgpufamilysupport") or "")

    @cached_property
    def total_memory_bytes(self) -> int:
        """Unified system memory visible to the integrated GPU."""
        return psutil.virtual_memory().total

    @property
    def mem_info(self) -> MemInfo:
        """Unified memory snapshot from the host OS."""
        vm = psutil.virtual_memory()
        return MemInfo(total_bytes=vm.total, used_bytes=vm.used, free_bytes=vm.available)

    @property
    def memory_readings(self) -> tuple[MemoryUsage, ...]:
        """Unified memory visible to CPU, GPU, and Neural Engine."""
        vm = psutil.virtual_memory()
        return (
            MemoryUsage(
                scope="unified",
                total_bytes=vm.total,
                used_bytes=vm.used,
                free_bytes=vm.available,
                unified=True,
                source="psutil",
            ),
        )

    @property
    def clock_readings(self) -> tuple[Clock, ...]:
        """Apple GPU clocks are not exposed without privileged sampling."""
        return (
            Clock(domain="gpu_compute", source="system_profiler", supported=False),
            Clock(domain="memory", source="system_profiler", supported=False),
        )


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
        hardware = apple_system_profile().get("SPHardwareDataType", [{}])[0]
        chip = hardware.get("chip_type") or "Apple Silicon"
        return f"{chip} Neural Engine"

    @cached_property
    def architecture(self) -> str:
        """Apple SoC family backing the Neural Engine."""
        hardware = apple_system_profile().get("SPHardwareDataType", [{}])[0]
        return str(hardware.get("chip_type") or "Apple Silicon")

    @cached_property
    def total_memory_bytes(self) -> int:
        """Unified system memory visible to the Neural Engine."""
        return psutil.virtual_memory().total

    @property
    def memory_readings(self) -> tuple[MemoryUsage, ...]:
        """Unified memory visible to CPU, GPU, and Neural Engine."""
        vm = psutil.virtual_memory()
        return (
            MemoryUsage(
                scope="unified",
                total_bytes=vm.total,
                used_bytes=vm.used,
                free_bytes=vm.available,
                unified=True,
                source="psutil",
            ),
        )

    @property
    def clock_readings(self) -> tuple[Clock, ...]:
        """Apple Neural Engine clocks are not exposed through public APIs."""
        return (Clock(domain="npu", source="coreml", supported=False),)
