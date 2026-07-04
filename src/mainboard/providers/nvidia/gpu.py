import logging
from contextlib import suppress
from functools import cached_property
from importlib import util

from pydantic import Field

from ...enums import CudaPythonVariant, Vendor
from ...gpu import GPU
from ...models.clock_info import ClockInfo
from ...models.compute_capability import ComputeCapability
from ...models.cuda_python_info import CudaPythonInfo
from ...models.energy_reading import EnergyReading
from ...models.memory import Memory
from ...models.pcie_info import PcieInfo
from ...models.process_info import ProcessInfo
from ...models.thermal_state import ThermalState
from ...models.utilization import Utilization
from . import apis
from .apis import NvidiaApis, text
from .protocols import CoreDevice, CoreSystem, DeviceProperties, NvmlHandle, SystemDevice

logger = logging.getLogger(__name__)


class NvidiaGPU(GPU):
    """NVIDIA CUDA device: static identity, build info, and live NVML sensors."""

    vendor: Vendor = Field(default=Vendor.NVIDIA)
    backend: str = "cuda"

    @classmethod
    def is_available(cls) -> bool:
        """Whether CUDA reports at least one NVIDIA device."""
        try:
            api = apis.nvidia_apis()
            err, count = api.runtime.cudaGetDeviceCount()
            return err == api.runtime.cudaError_t.cudaSuccess and count > 0
        except (ModuleNotFoundError, ImportError, OSError, RuntimeError):
            return False

    @classmethod
    def all(cls) -> tuple[NvidiaGPU, ...]:
        """Return all CUDA-visible devices ordered by visible index."""
        if not cls.is_available():
            return ()
        api = apis.nvidia_apis()
        _, count = api.runtime.cudaGetDeviceCount()
        return tuple(cls(index=i) for i in range(count))

    @cached_property
    def apis(self) -> NvidiaApis:
        """CUDA/NVML module handles."""
        return apis.nvidia_apis()

    @cached_property
    def cuda_device(self) -> CoreDevice:
        """Stable `cuda.core.Device` instance for this visible index.

        Only reached behind `has_cuda_core`, so the optional class is present here.
        """
        device_type = self.apis.cuda_device_type
        assert device_type is not None
        return device_type(self.index)

    @cached_property
    def system_api(self) -> CoreSystem:
        """The `cuda.core.system` module, present only behind `has_cuda_core`."""
        system = self.apis.system
        assert system is not None
        return system

    @cached_property
    def system_device(self) -> SystemDevice:
        """Stable `cuda.core.system.Device` instance for NVML-backed data.

        Only reached behind `has_cuda_core`, so the optional module is present here.
        """
        return self.system_api.Device(index=self.index)

    @cached_property
    def pci_bus_id(self) -> str:
        """PCI bus ID of the visible device, honoring `CUDA_VISIBLE_DEVICES`.

        Read through `cuda.bindings.runtime` so it works even when the
        optional `cuda.core` layer failed to import.
        """
        err, raw = self.apis.runtime.cudaDeviceGetPCIBusId(64, self.index)
        if err != self.apis.runtime.cudaError_t.cudaSuccess:
            raise RuntimeError(f"cudaDeviceGetPCIBusId({self.index}) failed: {err}")
        return text(raw).split("\x00", 1)[0].strip()

    @cached_property
    def handle(self) -> NvmlHandle:
        """NVML device handle resolved via PCI bus ID to respect `CUDA_VISIBLE_DEVICES`."""
        self.apis.nvml.init_v2()
        handle = self.apis.nvml.device_get_handle_by_pci_bus_id_v2(self.pci_bus_id)
        logger.debug(
            "GPU %s: %s (%s)",
            self.index,
            self.apis.nvml.device_get_name(handle),
            self.pci_bus_id,
        )
        return handle

    @cached_property
    def name(self) -> str:
        """Full GPU name string, e.g. `NVIDIA GeForce RTX 4090`."""
        if self.apis.has_cuda_core:
            return text(self.system_device.name)
        return text(self.apis.nvml.device_get_name(self.handle))

    @cached_property
    def uuid(self) -> str:
        """Unique NVIDIA GPU identifier."""
        if self.apis.has_cuda_core:
            return text(self.system_device.uuid)
        return text(self.apis.nvml.device_get_uuid(self.handle))

    @cached_property
    def cuda_architecture(self) -> ComputeCapability:
        """CUDA compute capability, e.g. `ComputeCapability(8, 9)`."""
        if self.apis.has_cuda_core:
            major, minor = self.system_device.cuda_compute_capability
        else:
            major, minor = self.apis.nvml.device_get_cuda_compute_capability(self.handle)
        return ComputeCapability(major, minor)

    @cached_property
    def runtime_properties(self) -> DeviceProperties:
        """Static device properties from `cuda.bindings.runtime`.

        ABI-stable source for SM count and bandwidth when the optional
        `cuda.core` layer is unavailable.
        """
        err, props = self.apis.runtime.cudaGetDeviceProperties(self.index)
        if err != self.apis.runtime.cudaError_t.cudaSuccess:
            raise RuntimeError(f"cudaGetDeviceProperties({self.index}) failed: {err}")
        return props

    @cached_property
    def architecture(self) -> str:
        """Human-readable NVIDIA architecture name, e.g. `Ada`."""
        if self.apis.has_cuda_core:
            arch = self.system_device.arch
            return str(getattr(arch, "name", arch)).title()
        return self.cuda_architecture.architecture

    @cached_property
    def arch_key(self) -> str:
        """The `sm_NN` compute-capability target, e.g. `sm_90` — the per-arch dispatch key."""
        return self.cuda_architecture.sm

    @cached_property
    def sm_count(self) -> int:
        """Number of streaming multiprocessors."""
        if self.apis.has_cuda_core:
            return self.cuda_device.properties.multiprocessor_count
        return self.runtime_properties.multiProcessorCount

    @cached_property
    def peak_bandwidth_gbs(self) -> float:
        """Theoretical peak memory bandwidth in GB/s."""
        if self.apis.has_cuda_core:
            props = self.cuda_device.properties
            return props.memory_clock_rate * 2 * props.global_memory_bus_width / 8 / 1e6
        bus_width = self.runtime_properties.memoryBusWidth
        max_mem_mhz = self.apis.nvml.device_get_max_clock_info(
            self.handle, self.apis.nvml.ClockType.CLOCK_MEM
        )
        return max_mem_mhz * 1e3 * 2 * bus_width / 8 / 1e6

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

    @cached_property
    def coherent(self) -> bool:
        """Whether this GPU shares a cache-coherent memory pool with the host.

        Probed, not guessed: a device that reports both
        `cudaDevAttrPageableMemoryAccess` and `cudaDevAttrConcurrentManagedAccess`
        sits on a coherent fabric where host RAM is a peer NUMA node of HBM (Grace
        Hopper, GB10), not a PCIe copy away. A discrete card (the 4090) reports
        neither, so `unified` stays False there. A binding that lacks the attribute
        query degrades to False rather than raising.
        """
        runtime = self.apis.runtime
        with suppress(*self.apis.nvml_errors, AttributeError):
            attrs = runtime.cudaDeviceAttr
            ok = runtime.cudaError_t.cudaSuccess
            err_p, pageable = runtime.cudaDeviceGetAttribute(
                attrs.cudaDevAttrPageableMemoryAccess, self.index
            )
            err_m, managed = runtime.cudaDeviceGetAttribute(
                attrs.cudaDevAttrConcurrentManagedAccess, self.index
            )
            return err_p == ok and err_m == ok and bool(pageable) and bool(managed)
        return False

    @property
    def memory(self) -> Memory:
        """CUDA-visible GPU memory allocation state.

        On GH200 and other coherent platforms this reflects HBM-resident
        allocations (the discrete-device counter) and carries `unified=True`, the
        probed signal that host RAM is a peer pool the residency policy can spend.
        Managed memory paged into Grace LPDDR is not counted here, matching
        `nvidia-smi`.
        """
        unified = self.coherent
        if self.apis.has_cuda_core:
            try:
                memory = self.system_device.memory_info
                return Memory(
                    scope="vram",
                    total_bytes=memory.total,
                    used_bytes=memory.used,
                    free_bytes=memory.free,
                    unified=unified,
                    source="cuda-core-system",
                )
            except self.system_api.NotSupportedError:
                return self.runtime_memory()
        return self.nvml_memory()

    def nvml_memory(self) -> Memory:
        """Current memory state from NVML when `cuda.core` is unavailable."""
        with suppress(*self.apis.nvml_errors):
            memory = self.apis.nvml.device_get_memory_info_v2(self.handle)
            return Memory(
                scope="vram",
                total_bytes=memory.total,
                used_bytes=memory.used,
                free_bytes=memory.free,
                unified=self.coherent,
                source="nvml",
            )
        return self.runtime_memory()

    def runtime_memory(self) -> Memory:
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
        return Memory(
            scope="vram",
            total_bytes=total_bytes,
            used_bytes=total_bytes - free_bytes,
            free_bytes=free_bytes,
            unified=self.coherent,
            source="cuda-runtime",
        )

    @property
    def clocks(self) -> ClockInfo:
        """Current SM and memory clock frequencies."""
        if not self.apis.has_cuda_core:
            return self.nvml_clocks()
        try:
            return ClockInfo(
                sm_mhz=self.system_device.get_clock("sm").get_current_mhz(),
                memory_mhz=self.system_device.get_clock("memory").get_current_mhz(),
            )
        except self.system_api.NotSupportedError:
            props = self.cuda_device.properties
            return ClockInfo(
                sm_mhz=round(props.clock_rate / 1000),
                memory_mhz=round(props.memory_clock_rate / 1000),
            )

    def nvml_clocks(self) -> ClockInfo:
        """Current clocks from NVML when `cuda.core` is unavailable."""
        clock = self.apis.nvml.ClockType
        return ClockInfo(
            sm_mhz=self.apis.nvml.device_get_clock_info(self.handle, clock.CLOCK_SM),
            memory_mhz=self.apis.nvml.device_get_clock_info(self.handle, clock.CLOCK_MEM),
        )

    @property
    def utilization(self) -> Utilization:
        """GPU core and memory-controller utilization percentages; zeros if unsupported."""
        if self.apis.has_cuda_core:
            try:
                utilization = self.system_device.utilization
            except self.system_api.NotSupportedError:
                return Utilization()
            return Utilization(gpu_pct=utilization.gpu, memory_pct=utilization.memory)
        with suppress(*self.apis.nvml_errors):
            rates = self.apis.nvml.device_get_utilization_rates(self.handle)
            return Utilization(gpu_pct=rates.gpu, memory_pct=rates.memory)
        return Utilization()

    @property
    def thermal(self) -> ThermalState:
        """Die temperature, thresholds, and throttle reasons; zeros where unsupported."""
        nvml = self.apis.nvml
        temperature_c = 0
        shutdown_threshold_c = 0
        slowdown_threshold_c = 0
        throttle_reasons = 0
        with suppress(*self.apis.nvml_errors):
            temperature_c = nvml.device_get_temperature_v(
                self.handle, nvml.TemperatureSensors.TEMPERATURE_GPU
            )
            shutdown_threshold_c = nvml.device_get_temperature_threshold(
                self.handle, nvml.TemperatureThresholds.TEMPERATURE_THRESHOLD_SHUTDOWN
            )
            slowdown_threshold_c = nvml.device_get_temperature_threshold(
                self.handle, nvml.TemperatureThresholds.TEMPERATURE_THRESHOLD_SLOWDOWN
            )
            throttle_reasons = nvml.device_get_current_clocks_event_reasons(self.handle)
        return ThermalState(
            temperature_c=temperature_c,
            shutdown_threshold_c=shutdown_threshold_c,
            slowdown_threshold_c=slowdown_threshold_c,
            throttle_reasons=throttle_reasons,
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
