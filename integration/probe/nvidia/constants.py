"""Stand-ins for `cuda.bindings`, with the shape its stubs in `stubs/` state, and the scripts a
fresh interpreter runs against them, printing JSON. They hold two devices, handle `100 + index`,
and a bus id `0000:0<index>:00.0` that the runtime and NVML both resolve."""

PACKAGES = {"cuda/__init__.py": "", "cuda/bindings/__init__.py": ""}

NVML = """
import enum


class NvmlError(Exception): ...
class GpuIsLostError(NvmlError): ...
class LibraryNotFoundError(NvmlError): ...
class NoPermissionError(NvmlError): ...
class NotSupportedError(NvmlError): ...
class UnknownError(NvmlError): ...


class ClockType(enum.IntEnum):
    CLOCK_SM = 1
    CLOCK_MEM = 2


class TemperatureSensors(enum.IntEnum):
    TEMPERATURE_GPU = 0


class ClocksEventReasons(enum.IntFlag):
    EVENT_REASON_SW_POWER_CAP = 4
    EVENT_REASON_SW_THERMAL_SLOWDOWN = 32
    EVENT_REASON_SYNC_BOOST = 16
    THROTTLE_REASON_HW_POWER_BRAKE_SLOWDOWN = 128
    THROTTLE_REASON_HW_SLOWDOWN = 8
    THROTTLE_REASON_HW_THERMAL_SLOWDOWN = 64


class PciInfo:
    def __init__(self, handle):
        self.bus_id = f"0000:0{handle - 100}:00.0"


def init_v2(): ...
def device_get_count_v2(): return 2
def device_get_handle_by_index_v2(index): return 100 + index
def device_get_handle_by_pci_bus_id_v2(pci_bus_id): return 100 + int(pci_bus_id[6])
def device_get_handle_by_uuid(uuid): return int(uuid.removeprefix("GPU-"))
def device_get_name(handle): return f"Fake GPU {handle}"
def device_get_pci_info_v3(handle): return PciInfo(handle)
def device_get_cuda_compute_capability(handle): return (8, 9)
def device_get_current_clocks_event_reasons(handle): return 4 | 8 | 1
def device_get_temperature_v(handle, sensor): return 61
def system_get_driver_version(): return "999.1"
"""

RUNTIME = """
import enum


class cudaError_t(enum.IntEnum):
    cudaSuccess = 0


class cudaDeviceAttr(enum.IntEnum):
    cudaDevAttrConcurrentManagedAccess = 2
    cudaDevAttrHostNativeAtomicSupported = 3
    cudaDevAttrPageableMemoryAccess = 1


def cudaDeviceGetAttribute(attr, device): return (cudaError_t.cudaSuccess, 1)
def cudaDeviceGetPCIBusId(length, device):
    return (cudaError_t.cudaSuccess, f"0000:0{device}:00.0\\x00".encode())
def cudaGetDeviceCount(): return (cudaError_t.cudaSuccess, 2)
def cudaRuntimeGetVersion(): return (cudaError_t.cudaSuccess, 13030)
"""

READINGS = """
import json
import os

from mainboard.probe import NvidiaGPU

os.environ.pop("CUDA_VISIBLE_DEVICES", None)
gpus = NvidiaGPU.all()
print(json.dumps({
    "labels": [gpu.label for gpu in gpus],
    "bus_ids": [gpu.pci_bus_id for gpu in gpus],
    "arch": [gpu.arch_key for gpu in gpus],
    "coherent": [gpu.coherent for gpu in gpus],
    "runtime": [gpu.runtime_version for gpu in gpus],
    "driver": gpus[0].driver,
    "temperature": gpus[0].temperature_c(),
    "throttles": gpus[0].throttles(),
}))
"""

MASKED_READINGS = """
import json
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from mainboard.probe import NvidiaGPU

gpus = NvidiaGPU.all()
print(json.dumps({
    "labels": [gpu.label for gpu in gpus],
    "bus_ids": [gpu.pci_bus_id for gpu in gpus],
    "runtime": [gpu.runtime_version for gpu in gpus],
}))
"""

ABSENT = """
import json

from mainboard.probe import NvidiaGPU

print(json.dumps([NvidiaGPU.is_available(), len(NvidiaGPU.all())]))
"""
