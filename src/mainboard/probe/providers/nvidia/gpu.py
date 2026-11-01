import os
from contextlib import suppress
from functools import cached_property
from itertools import takewhile

from ....core.constants.units import BYTES_PER_GB
from ....log import logger
from ...enums import Vendor
from ...facts.devices.telemetry import Energy, Telemetry, Thermal, UnitProcess
from ...facts.devices.utilization import Utilization
from ...facts.memory.core import Memory
from ...units.gpu import GPU
from . import apis
from .capability import ComputeCapability
from .memory import Vram

_HZ_PER_MHZ = 1e6
_KHZ_PER_MHZ = 1000
_MILLIWATTS_PER_WATT = 1000.0


def _bus_key(bus_id: str) -> str:
    """A PCI bus id with its domain normalized to four hex digits.

    The runtime prints four digits of domain (`0000:21:00.0`) and the system layer eight
    (`00000000:21:00.0`).
    """
    domain, _, rest = bus_id.strip().lower().partition(":")
    return f"{int(domain or '0', 16):04x}:{rest}"


def _text(value: bytes | str) -> str:
    """Convert CUDA/NVML byte strings and scalars to text."""
    return value.decode(errors="replace") if isinstance(value, bytes) else value


def _visible_devices() -> list[str] | None:
    """The entries of `CUDA_VISIBLE_DEVICES`, or None when the mask is unset.

    CUDA reads the list up to its first entry that names no device (an index or a `GPU-`/`MIG-`
    UUID), so the list is cut there the same way; an empty mask hides every device.
    """
    if (mask := os.environ.get("CUDA_VISIBLE_DEVICES")) is None:
        return None
    return list(
        takewhile(
            lambda entry: entry.startswith(("GPU-", "MIG-")) or entry.lstrip("-").isdigit(),
            (entry.strip() for entry in mask.split(",")),
        )
    )


class NvidiaGPU(GPU):
    """NVIDIA CUDA device with static identity, capability, and memory capacity."""

    vendor: Vendor = Vendor.NVIDIA
    backend: str = "cuda"

    @cached_property
    def arch_key(self) -> str:
        """The `sm_NN` compute-capability target, e.g. `sm_90`, the per-arch dispatch key."""
        return self.cuda_architecture.sm

    @cached_property
    def architecture(self) -> str:
        """Human-readable NVIDIA architecture name, e.g. `Ada`."""
        if apis.has_cuda_core:
            return self.system_device.arch.name.title()
        return self.cuda_architecture.architecture

    @cached_property
    def coherent(self) -> bool:
        """Whether this GPU shares a cache-coherent memory pool with the host.

        Probed on three attributes, not two: a driver with heterogeneous memory management
        makes a discrete card report `cudaDevAttrPageableMemoryAccess` and
        `cudaDevAttrConcurrentManagedAccess` too (the RTX 4090 on the open kernel modules did,
        and was budgeted as a Grace Hopper for a week), while
        `cudaDevAttrHostNativeAtomicSupported` holds only where host RAM is a peer of HBM over
        a coherent fabric (GH200, GB10). A binding lacking the query degrades to False.
        """
        if not apis.has_runtime:
            return False
        with suppress(*apis.nvml_errors, AttributeError):
            attrs = apis.runtime.cudaDeviceAttr
            answers = [
                apis.runtime.cudaDeviceGetAttribute(attr, self.index)
                for attr in (
                    attrs.cudaDevAttrPageableMemoryAccess,
                    attrs.cudaDevAttrConcurrentManagedAccess,
                    attrs.cudaDevAttrHostNativeAtomicSupported,
                )
            ]
            success = apis.runtime.cudaError_t.cudaSuccess
            return all(err == success and bool(value) for err, value in answers)
        return False

    @cached_property
    def cuda_architecture(self) -> ComputeCapability:
        """CUDA compute capability, e.g. `ComputeCapability(8, 9)`."""
        if apis.has_cuda_core:
            major, minor = self.system_device.cuda_compute_capability
        else:
            major, minor = apis.nvml.device_get_cuda_compute_capability(self.handle)
        return ComputeCapability(major, minor)

    @cached_property
    def driver(self) -> str:
        """The installed NVIDIA driver, as `nvidia-smi` prints it, or empty when NVML refuses."""
        apis.nvml.init_v2()
        with suppress(*apis.nvml_errors, AttributeError):
            return _text(apis.nvml.system_get_driver_version())
        return ""

    @cached_property
    def handle(self) -> int:
        """NVML device handle resolved via PCI bus ID to respect `CUDA_VISIBLE_DEVICES`."""
        apis.nvml.init_v2()
        if apis.has_runtime:
            bus_id = self.pci_bus_id
            handle = apis.nvml.device_get_handle_by_pci_bus_id_v2(bus_id)
        else:
            handle = self.__nvml_handle()
            bus_id = _text(apis.nvml.device_get_pci_info_v3(handle).bus_id)
        logger.debug("GPU {}: {} ({})", self.index, apis.nvml.device_get_name(handle), bus_id)
        return handle

    @cached_property
    def label(self) -> str:
        """Full GPU name string, e.g. `NVIDIA GeForce RTX 4090`."""
        if apis.has_cuda_core:
            return _text(self.system_device.name)
        return _text(apis.nvml.device_get_name(self.handle))

    @cached_property
    def pci_bus_id(self) -> str:
        """PCI bus ID of the visible device, honoring `CUDA_VISIBLE_DEVICES`.

        The CUDA Runtime preserves visible-device remapping; a pure-NVML stack reads the same
        identity from the physical handle instead.
        """
        if not apis.has_runtime:
            return _text(apis.nvml.device_get_pci_info_v3(self.handle).bus_id)
        err, raw = apis.runtime.cudaDeviceGetPCIBusId(64, self.index)
        if err != apis.runtime.cudaError_t.cudaSuccess:
            raise RuntimeError(f"cudaDeviceGetPCIBusId({self.index}) failed: {err}")
        return _text(raw).split("\x00", 1)[0].strip()

    @cached_property
    def peak_bandwidth_gbs(self) -> float:
        """Theoretical peak memory bandwidth in GB/s, 0.0 when NVML will not report it.

        The bus width times the maximum memory clock, doubled because the memories move data
        on both clock edges: exactly the vendor's 1008 GB/s for a 4090's 384-bit bus at 10501 MHz.
        """
        with suppress(*apis.nvml_errors):
            nvml = apis.nvml
            clock_mhz = nvml.device_get_max_clock_info(self.handle, nvml.ClockType.CLOCK_MEM)
            bus_bytes = nvml.device_get_memory_bus_width(self.handle) / 8
            return clock_mhz * _HZ_PER_MHZ * 2 * bus_bytes / BYTES_PER_GB
        return 0.0

    @cached_property
    def peak_clock_khz(self) -> int:
        """The maximum SM clock NVML reports, in kHz; 0 when NVML cannot say."""
        with suppress(*apis.nvml_errors):
            nvml = apis.nvml
            return (
                nvml.device_get_max_clock_info(self.handle, nvml.ClockType.CLOCK_SM) * _KHZ_PER_MHZ
            )
        return 0

    @cached_property
    def runtime_version(self) -> tuple[int, int] | None:
        """The CUDA runtime version this process links, e.g. `(13, 3)`."""
        if not apis.has_runtime:
            return None
        _, raw = apis.runtime.cudaRuntimeGetVersion()
        return (raw // _KHZ_PER_MHZ, (raw % _KHZ_PER_MHZ) // 10)

    @cached_property
    def system_device(self) -> apis.system.Device:
        """Stable `cuda.core.system.Device` instance for NVML-backed data.

        The system layer enumerates physical devices and ignores `CUDA_VISIBLE_DEVICES`, so the
        device is picked by the PCI bus id the runtime resolves for this visible index; a job
        pinned to the idle second card of a shared box read the first card's load before this.
        """
        bus_id = _bus_key(self.pci_bus_id)
        for candidate in apis.system.Device.get_all_devices():
            if _bus_key(_text(candidate.pci_bus_id)) == bus_id:
                return candidate
        return apis.system.Device(index=self.index)

    @cached_property
    def uuid(self) -> str:
        """Unique NVIDIA GPU identifier."""
        if apis.has_cuda_core:
            return _text(self.system_device.uuid)
        return _text(apis.nvml.device_get_uuid(self.handle))

    @classmethod
    def all(cls) -> tuple[NvidiaGPU, ...]:
        """All CUDA-visible devices ordered by visible index."""
        if not cls.is_available():
            return ()
        return tuple(cls(index=i) for i in range(cls.device_count()))

    @classmethod
    def device_count(cls) -> int:
        """How many devices the runtime or NVML reports, 0 when its query failed."""
        if apis.has_runtime:
            err, count = apis.runtime.cudaGetDeviceCount()
            return count if err == apis.runtime.cudaError_t.cudaSuccess else 0
        apis.nvml.init_v2()
        count = apis.nvml.device_get_count_v2()
        visible = _visible_devices()
        return count if visible is None else min(len(visible), count)

    @classmethod
    def is_available(cls) -> bool:
        """Whether CUDA reports at least one NVIDIA device."""
        if not apis.has_nvml:
            return False
        faults: tuple[type[Exception], ...] = (*apis.nvml_errors, OSError, RuntimeError)
        try:
            return cls.device_count() > 0
        except faults:
            return False

    @property
    def memory(self) -> Memory:
        """CUDA-visible GPU memory, read by `Vram` through the first layer that answers."""
        return Vram(self).read()

    @property
    def utilization(self) -> Utilization:
        """Current compute and memory-controller utilization."""
        if apis.has_cuda_core:
            with suppress(apis.system.NotSupportedError):
                reading = self.system_device.utilization
                return Utilization(gpu_pct=reading.gpu, memory_pct=reading.memory)
        with suppress(*apis.nvml_errors):
            rates = apis.nvml.device_get_utilization_rates(self.handle)
            return Utilization(gpu_pct=rates.gpu, memory_pct=rates.memory)
        return Utilization()

    def __nvml_handle(self) -> int:
        """The NVML handle of this visible index under the same mask CUDA applies.

        Without a runtime to remap for it, NVML enumerates every physical device, so the
        visible index is read through `CUDA_VISIBLE_DEVICES` first. A job pinned to the idle
        second card of a shared box was judged by the first card's load before this, and its
        idle gate refused it.
        """
        if (visible := _visible_devices()) is None:
            return apis.nvml.device_get_handle_by_index_v2(self.index)
        entry = visible[self.index]
        if entry.startswith(("GPU-", "MIG-")):
            return apis.nvml.device_get_handle_by_uuid(entry)
        return apis.nvml.device_get_handle_by_index_v2(int(entry))

    def power_w(self) -> float:
        """Instantaneous power draw in watts, 0.0 when NVML will not report it."""
        with suppress(*apis.nvml_errors):
            return apis.nvml.device_get_power_usage(self.handle) / _MILLIWATTS_PER_WATT
        return 0.0

    def processes(self) -> tuple[UnitProcess, ...]:
        """Every compute process on the card, empty when NVML cannot say."""
        with suppress(*apis.nvml_errors):
            running = apis.nvml.device_get_compute_running_processes_v3(self.handle)
            return tuple(
                UnitProcess(pid=item.pid, used_bytes=item.used_gpu_memory) for item in running
            )
        return ()

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
        nvml = apis.nvml
        with suppress(*apis.nvml_errors):
            return nvml.device_get_temperature_v(
                self.handle, nvml.TemperatureSensors.TEMPERATURE_GPU
            )
        return 0

    def throttles(self) -> tuple[str, ...]:
        """The real slowdowns NVML reports as active, ignoring the benign clock states.

        NVML answers with one bitmask covering both, and an idle device always sets a bit,
        so reading the mask as a boolean would report every idle GPU as throttled.
        """
        nvml = apis.nvml
        with suppress(*apis.nvml_errors):
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
