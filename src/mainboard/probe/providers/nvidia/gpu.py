import logging
from contextlib import suppress
from functools import cached_property

from ...enums import Vendor
from ...facts.memory import Memory
from ...facts.telemetry import Energy, Telemetry, Thermal, UnitProcess
from ...facts.utilization import Utilization
from ...units.gpu import GPU
from . import apis
from .apis import NvidiaApis, text
from .capability import ComputeCapability
from .protocols import (  # ruff: ignore[typing-only-first-party-import] reason=keeps protocols.py's runtime import exercised for coverage since=2026-08-17
    CoreSystem,
    NvmlHandle,
    SystemDevice,
)

logger = logging.getLogger(__name__)


class NvidiaGPU(GPU):
    """NVIDIA CUDA device with static identity, capability, and memory capacity."""

    vendor: Vendor = Vendor.NVIDIA
    backend: str = "cuda"

    @cached_property
    def apis(self) -> NvidiaApis:
        """CUDA/NVML module handles."""
        return apis.nvidia_apis()

    @cached_property
    def arch_key(self) -> str:
        """The `sm_NN` compute-capability target, e.g. `sm_90`, the per-arch dispatch key."""
        return self.cuda_architecture.sm

    @cached_property
    def architecture(self) -> str:
        """Human-readable NVIDIA architecture name, e.g. `Ada`."""
        if self.apis.has_cuda_core:
            arch = self.system_device.arch
            return str(getattr(arch, "name", arch)).title()
        return self.cuda_architecture.architecture

    @cached_property
    def coherent(self) -> bool:
        """Whether this GPU shares a cache-coherent memory pool with the host.

        This is probed rather than guessed, since a device that reports both
        `cudaDevAttrPageableMemoryAccess` and `cudaDevAttrConcurrentManagedAccess`
        sits on a coherent fabric where host RAM is a peer NUMA node of HBM (Grace
        Hopper, GB10), not a PCIe copy away. A discrete card (the 4090) reports
        neither, so `unified` stays False there. A binding that lacks the attribute
        query degrades to False rather than raising.
        """
        runtime = self.apis.runtime
        with suppress(*self.apis.nvml_errors, AttributeError):
            attrs = runtime.cudaDeviceAttr
            success = runtime.cudaError_t.cudaSuccess
            err_p, pageable = runtime.cudaDeviceGetAttribute(
                attrs.cudaDevAttrPageableMemoryAccess, self.index
            )
            err_m, managed = runtime.cudaDeviceGetAttribute(
                attrs.cudaDevAttrConcurrentManagedAccess, self.index
            )
            return err_p == success and err_m == success and bool(pageable) and bool(managed)
        return False

    @cached_property
    def cuda_architecture(self) -> ComputeCapability:
        """CUDA compute capability, e.g. `ComputeCapability(8, 9)`."""
        if self.apis.has_cuda_core:
            major, minor = self.system_device.cuda_compute_capability
        else:
            major, minor = self.apis.nvml.device_get_cuda_compute_capability(self.handle)
        return ComputeCapability(major, minor)

    @cached_property
    def driver_version(self) -> tuple[int, int]:
        """Maximum CUDA version supported by the installed driver."""
        _, raw = self.apis.runtime.cudaDriverGetVersion()
        return (raw // 1000, (raw % 1000) // 10)

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
    def label(self) -> str:
        """Full GPU name string, e.g. `NVIDIA GeForce RTX 4090`."""
        if self.apis.has_cuda_core:
            return text(self.system_device.name)
        return text(self.apis.nvml.device_get_name(self.handle))

    @property
    def memory(self) -> Memory:
        """CUDA-visible GPU memory allocation state.

        A three-tier fallback, `cuda.core.system` when it loaded, NVML next, and the
        CUDA Runtime's `cudaMemGetInfo` as the last resort when NVML memory is
        unsupported. On GH200 and other coherent platforms this reflects HBM-resident
        allocations (the discrete-device counter) and carries `unified=True`, the
        probed signal that host RAM is a peer pool. Managed memory paged into Grace
        LPDDR is not counted here, matching `nvidia-smi`.
        """
        unified = self.coherent
        if self.apis.has_cuda_core:
            try:
                memory = self.system_device.memory_info
            except self.system_api.NotSupportedError:
                return self.runtime_memory()
            return Memory(
                scope="vram",
                total_bytes=memory.total,
                used_bytes=memory.used,
                free_bytes=memory.free,
                unified=unified,
                source="cuda-core-system",
            )
        return self.nvml_memory()

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
    def peak_bandwidth_gbs(self) -> float:
        """Theoretical peak memory bandwidth in GB/s, 0.0 when NVML will not report it.

        The bus width times the maximum memory clock, doubled because the memories move
        data on both clock edges. That reproduces the vendor headline figure exactly, 1008
        GB/s for a 4090's 384-bit bus at 10501 MHz.
        """
        with suppress(*self.apis.nvml_errors):
            nvml = self.apis.nvml
            clock_mhz = nvml.device_get_max_clock_info(self.handle, nvml.ClockType.CLOCK_MEM)
            bus_bytes = nvml.device_get_memory_bus_width(self.handle) / 8
            return clock_mhz * 1e6 * 2 * bus_bytes / 1e9
        return 0.0

    @cached_property
    def system_api(self) -> CoreSystem:
        """The `cuda.core.system` module, present only behind `has_cuda_core`."""
        if self.apis.system is None:
            raise RuntimeError("cuda.core.system is unavailable, check has_cuda_core first")
        return self.apis.system

    @cached_property
    def system_device(self) -> SystemDevice:
        """Stable `cuda.core.system.Device` instance for NVML-backed data.

        Only reached behind `has_cuda_core`, so the optional module is present here.
        """
        return self.system_api.Device(index=self.index)

    @property
    def utilization(self) -> Utilization:
        """Current compute and memory-controller utilization."""
        if self.apis.has_cuda_core:
            with suppress(self.system_api.NotSupportedError):
                reading = self.system_device.utilization
                return Utilization(gpu_pct=reading.gpu, memory_pct=reading.memory)
        with suppress(*self.apis.nvml_errors):
            reading = self.apis.nvml.device_get_utilization_rates(self.handle)
            return Utilization(gpu_pct=reading.gpu, memory_pct=reading.memory)
        return Utilization()

    @cached_property
    def uuid(self) -> str:
        """Unique NVIDIA GPU identifier."""
        if self.apis.has_cuda_core:
            return text(self.system_device.uuid)
        return text(self.apis.nvml.device_get_uuid(self.handle))

    @classmethod
    def all(cls) -> tuple[NvidiaGPU, ...]:
        """Return all CUDA-visible devices ordered by visible index."""
        if not cls.is_available():
            return ()
        api = apis.nvidia_apis()
        _, count = api.runtime.cudaGetDeviceCount()
        return tuple(cls(index=i) for i in range(count))

    @classmethod
    def device_count(cls) -> int:
        """How many devices CUDA reports, 0 when the count call itself failed."""
        api = apis.nvidia_apis()
        err, count = api.runtime.cudaGetDeviceCount()
        return count if err == api.runtime.cudaError_t.cudaSuccess else 0

    @classmethod
    def is_available(cls) -> bool:
        """Whether CUDA reports at least one NVIDIA device."""
        try:
            return cls.device_count() > 0
        except ModuleNotFoundError, ImportError, OSError, RuntimeError:
            return False

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

    def power_w(self) -> float:
        """Instantaneous power draw in watts, 0.0 when NVML will not report it."""
        with suppress(*self.apis.nvml_errors):
            return self.apis.nvml.device_get_power_usage(self.handle) / 1000.0
        return 0.0

    def processes(self) -> tuple[UnitProcess, ...]:
        """Every process holding a compute context here, empty when NVML will not say."""
        with suppress(*self.apis.nvml_errors):
            running = self.apis.nvml.device_get_compute_running_processes_v3(self.handle)
            return tuple(
                UnitProcess(pid=item.pid, used_bytes=item.used_gpu_memory) for item in running
            )
        return ()

    def runtime_memory(self) -> Memory:
        """Current memory state from CUDA Runtime when NVML memory is unsupported."""
        err, current = self.apis.runtime.cudaGetDevice()
        if err != self.apis.runtime.cudaError_t.cudaSuccess:
            current = self.index
        self.apis.runtime.cudaSetDevice(self.index)
        try:
            err, free_bytes, total_bytes = self.apis.runtime.cudaMemGetInfo()
        finally:
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

    def snapshot(self, name: str = "") -> Telemetry:
        """Point-in-time NVML reading of this device's sensors, tagged with region `name`.

        Each sensor degrades on its own, so a device that reports power but refuses
        per-process memory still returns the power.
        """
        return Telemetry(
            unit_name=self.label,
            region=name,
            energy=Energy(power_w=self.power_w()),
            thermal=Thermal(temperature_c=self.temperature_c(), throttle_names=self.throttles()),
            utilization=self.utilization,
            processes=self.processes(),
        )

    def temperature_c(self) -> int:
        """Die temperature in degrees Celsius, 0 when NVML will not report it."""
        nvml = self.apis.nvml
        with suppress(*self.apis.nvml_errors):
            sensor = nvml.TemperatureSensors.TEMPERATURE_GPU
            return nvml.device_get_temperature_v(self.handle, sensor)
        return 0

    def throttles(self) -> tuple[str, ...]:
        """The real slowdowns NVML reports as active, ignoring the benign clock states.

        NVML answers with one bitmask covering both, and an idle device always sets a bit,
        so reading the mask as a boolean would report every idle GPU as throttled.
        """
        nvml = self.apis.nvml
        with suppress(*self.apis.nvml_errors):
            active = nvml.device_get_current_clocks_event_reasons(self.handle)
            reasons = nvml.ClocksEventReasons
            slowdowns = {
                "power cap": reasons.EVENT_REASON_SW_POWER_CAP,
                "thermal": reasons.EVENT_REASON_SW_THERMAL_SLOWDOWN,
                "sync boost": reasons.EVENT_REASON_SYNC_BOOST,
                "hardware power brake": reasons.THROTTLE_REASON_HW_POWER_BRAKE_SLOWDOWN,
                "hardware slowdown": reasons.THROTTLE_REASON_HW_SLOWDOWN,
                "hardware thermal": reasons.THROTTLE_REASON_HW_THERMAL_SLOWDOWN,
            }
            return tuple(label for label, bit in slowdowns.items() if active & bit)
        return ()
