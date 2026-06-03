from __future__ import annotations

import logging
from contextlib import suppress
from functools import cache, cached_property
from importlib import import_module, util
from typing import Any

from pydantic import Field

from ..enums import CudaPythonVariant, Vendor
from ..gpu import GPU
from ..models.clock_info import ClockInfo
from ..models.compute_capability import ComputeCapability
from ..models.cuda_python_info import CudaPythonInfo
from ..models.energy_reading import EnergyReading
from ..models.mem_info import MemInfo
from ..models.memory_usage import MemoryUsage
from ..models.pcie_info import PcieInfo
from ..models.process_info import ProcessInfo
from ..models.thermal_state import ThermalState
from ..models.utilization import Utilization

logger = logging.getLogger(__name__)


def text(value: Any) -> str:
    """Convert CUDA/NVML byte strings and scalars to text."""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return str(value)


class NvidiaApis:
    """Lazy CUDA/NVML imports for the NVIDIA provider."""

    def __init__(self) -> None:
        self.runtime: Any = import_module("cuda.bindings.runtime")
        self.system: Any = import_module("cuda.core.system")
        try:
            self.nvml: Any = import_module("cuda.bindings._nvml")
        except ModuleNotFoundError:
            self.nvml = import_module("cuda.bindings.nvml")
        cuda_core: Any = import_module("cuda.core")
        self.cuda_device_type: Any = cuda_core.Device
        self.nvml_errors: tuple[type[Exception], ...] = tuple(
            error
            for name in (
                "NotSupportedError",
                "NoPermissionError",
                "UnknownError",
                "GpuIsLostError",
            )
            if isinstance(error := getattr(self.nvml, name, None), type)
            and issubclass(error, Exception)
        )


@cache
def nvidia_apis() -> NvidiaApis:
    """Return cached CUDA/NVML imports for NVIDIA devices."""
    return NvidiaApis()


class NvidiaGPU(GPU):
    """NVIDIA CUDA device: static identity, build info, and live NVML sensors."""

    vendor: Vendor = Field(default=Vendor.NVIDIA)
    backend: str = "cuda"

    @classmethod
    def is_available(cls) -> bool:
        """Whether CUDA reports at least one NVIDIA device."""
        try:
            api = nvidia_apis()
            err, count = api.runtime.cudaGetDeviceCount()
            return err == api.runtime.cudaError_t.cudaSuccess and count > 0
        except (ModuleNotFoundError, ImportError, OSError, RuntimeError):
            return False

    @classmethod
    def all(cls) -> tuple[NvidiaGPU, ...]:
        """Return all CUDA-visible devices ordered by visible index."""
        if not cls.is_available():
            return ()
        api = nvidia_apis()
        _, count = api.runtime.cudaGetDeviceCount()
        return tuple(cls(index=i) for i in range(count))

    @cached_property
    def apis(self) -> NvidiaApis:
        """CUDA/NVML module handles."""
        return nvidia_apis()

    @cached_property
    def cuda_device(self) -> Any:
        """Stable `cuda.core.Device` instance for this visible index."""
        return self.apis.cuda_device_type(self.index)

    @cached_property
    def system_device(self) -> Any:
        """Stable `cuda.core.system.Device` instance for NVML-backed data."""
        return self.apis.system.Device(index=self.index)

    @cached_property
    def runtime_props(self) -> Any:
        """CUDA Runtime device properties."""
        err, props = self.apis.runtime.cudaGetDeviceProperties(self.index)
        if err != self.apis.runtime.cudaError_t.cudaSuccess:
            raise RuntimeError(f"cudaGetDeviceProperties({self.index}) failed: {err}")
        return props

    @cached_property
    def handle(self) -> Any:
        """NVML device handle resolved via PCI bus ID to respect `CUDA_VISIBLE_DEVICES`."""
        self.apis.nvml.init_v2()
        handle = self.apis.nvml.device_get_handle_by_pci_bus_id_v2(self.cuda_device.pci_bus_id)
        logger.debug(
            "GPU %s: %s (%s)",
            self.index,
            self.apis.nvml.device_get_name(handle),
            self.cuda_device.pci_bus_id,
        )
        return handle

    @cached_property
    def name(self) -> str:
        """Full GPU name string, e.g. `NVIDIA GeForce RTX 4090`."""
        return text(self.system_device.name)

    @cached_property
    def uuid(self) -> str:
        """Unique NVIDIA GPU identifier."""
        return text(self.system_device.uuid)

    @cached_property
    def cuda_architecture(self) -> ComputeCapability:
        """CUDA compute capability, e.g. `ComputeCapability(8, 9)`."""
        major, minor = self.system_device.cuda_compute_capability
        return ComputeCapability(major, minor)

    @cached_property
    def architecture(self) -> str:
        """Human-readable NVIDIA architecture name, e.g. `Ada`."""
        arch = self.system_device.arch
        return str(getattr(arch, "name", arch)).title()

    @cached_property
    def sm_count(self) -> int:
        """Number of streaming multiprocessors."""
        return self.cuda_device.properties.multiprocessor_count

    @cached_property
    def total_memory_bytes(self) -> int:
        """Total CUDA-visible memory in bytes."""
        return self.runtime_props.totalGlobalMem

    @cached_property
    def peak_bandwidth_gbs(self) -> float:
        """Theoretical peak memory bandwidth in GB/s."""
        props = self.cuda_device.properties
        return props.memory_clock_rate * 2 * props.global_memory_bus_width / 8 / 1e6

    @cached_property
    def driver_version(self) -> tuple[int, int]:
        """Maximum CUDA version supported by the installed driver."""
        _, raw = self.apis.runtime.cudaDriverGetVersion()
        return (raw // 1000, (raw % 1000) // 10)

    @cached_property
    def cuda_python(self) -> CudaPythonInfo:
        """Detected CUDA Python stack variant and CUPTI availability."""
        variant = (
            CudaPythonVariant.CU13 if self.driver_version[0] >= 13 else CudaPythonVariant.CU12
        )
        return CudaPythonInfo(
            variant=variant,
            supports_cupti=variant == CudaPythonVariant.CU13
            and util.find_spec("cupti") is not None,
        )

    @property
    def mem_info(self) -> MemInfo:
        """Current GPU memory allocation state."""
        try:
            memory = self.system_device.memory_info
            return MemInfo(
                total_bytes=memory.total,
                used_bytes=memory.used,
                free_bytes=memory.free,
            )
        except self.apis.system.NotSupportedError:
            return self.runtime_mem_info()

    @property
    def memory_readings(self) -> tuple[MemoryUsage, ...]:
        """CUDA-visible GPU memory."""
        mem = self.mem_info
        return (
            MemoryUsage(
                scope="vram",
                total_bytes=mem.total_bytes,
                used_bytes=mem.used_bytes,
                free_bytes=mem.free_bytes,
                source="cuda-runtime" if mem.used_bytes else "cuda-core-system",
            ),
        )

    def runtime_mem_info(self) -> MemInfo:
        """Current memory state from CUDA Runtime when NVML memory is unsupported."""
        err, current = self.apis.runtime.cudaGetDevice()
        has_current = err == self.apis.runtime.cudaError_t.cudaSuccess
        self.apis.runtime.cudaSetDevice(self.index)
        try:
            err, free_bytes, total_bytes = self.apis.runtime.cudaMemGetInfo()
        finally:
            if has_current:
                self.apis.runtime.cudaSetDevice(current)
        if err != self.apis.runtime.cudaError_t.cudaSuccess:
            raise RuntimeError(f"cudaMemGetInfo({self.index}) failed: {err}")
        return MemInfo(
            total_bytes=total_bytes,
            used_bytes=total_bytes - free_bytes,
            free_bytes=free_bytes,
        )

    @property
    def clocks(self) -> ClockInfo:
        """Current SM and memory clock frequencies."""
        try:
            return ClockInfo(
                sm_mhz=self.system_device.get_clock("sm").get_current_mhz(),
                memory_mhz=self.system_device.get_clock("memory").get_current_mhz(),
            )
        except self.apis.system.NotSupportedError:
            props = self.cuda_device.properties
            return ClockInfo(
                sm_mhz=round(props.clock_rate / 1000),
                memory_mhz=round(props.memory_clock_rate / 1000),
            )

    @property
    def utilization(self) -> Utilization:
        """GPU core and memory-controller utilization percentages."""
        utilization = self.system_device.utilization
        return Utilization(gpu_pct=utilization.gpu, memory_pct=utilization.memory)

    @property
    def thermal(self) -> ThermalState:
        """Die temperature, shutdown/slowdown thresholds, and throttle reasons."""
        throttle = self.apis.nvml.device_get_current_clocks_event_reasons(self.handle)
        return ThermalState(
            temperature_c=self.apis.nvml.device_get_temperature_v(
                self.handle,
                self.apis.nvml.TemperatureSensors.TEMPERATURE_GPU,
            ),
            shutdown_threshold_c=self.apis.nvml.device_get_temperature_threshold(
                self.handle,
                self.apis.nvml.TemperatureThresholds.TEMPERATURE_THRESHOLD_SHUTDOWN,
            ),
            slowdown_threshold_c=self.apis.nvml.device_get_temperature_threshold(
                self.handle,
                self.apis.nvml.TemperatureThresholds.TEMPERATURE_THRESHOLD_SLOWDOWN,
            ),
            throttle_reasons=throttle,
        )

    @property
    def energy(self) -> EnergyReading:
        """Current power draw and cumulative energy; zeros if unsupported."""
        power_mw = 0
        energy_mj = 0
        with suppress(*self.apis.nvml_errors):
            power_mw = self.apis.nvml.device_get_power_usage(self.handle)
            energy_mj = self.apis.nvml.device_get_total_energy_consumption(self.handle)
        return EnergyReading(power_mw=power_mw, energy_mj=energy_mj)

    @property
    def pcie(self) -> PcieInfo:
        """PCIe TX/RX throughput in KB/s; zeros on non-PCIe devices."""
        tx_kbs = 0
        rx_kbs = 0
        with suppress(*self.apis.nvml_errors):
            tx_kbs = self.apis.nvml.device_get_pcie_throughput(self.handle, 0)
            rx_kbs = self.apis.nvml.device_get_pcie_throughput(self.handle, 1)
        return PcieInfo(tx_kbs=tx_kbs, rx_kbs=rx_kbs)

    @property
    def fan_speed_pct(self) -> int:
        """Fan speed as a percentage; 0 on fanless devices."""
        with suppress(*self.apis.nvml_errors):
            return self.apis.nvml.device_get_fan_speed(self.handle)
        return 0

    @property
    def processes(self) -> list[ProcessInfo]:
        """Running compute processes on this GPU."""
        with suppress(*self.apis.nvml_errors):
            raw = self.apis.nvml.device_get_compute_running_processes_v3(self.handle)
            return [
                ProcessInfo(pid=raw[i].pid, used_bytes=raw[i].used_gpu_memory)
                for i in range(len(raw))
            ]
        return []
