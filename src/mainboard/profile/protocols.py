# Structural contracts profiling reads through, so nothing here or reading a finished `Profile`
# names a vendor backend.

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Sequence

# JSON values accepted by the Chrome/Perfetto trace-event writer.
type Json = str | int | float | bool | list[Json] | dict[str, Json] | None
type TraceEvent = dict[str, Json]


class TimedActivity(Protocol):
    """The fields every CUPTI record carries: the runtime `kind` and device-clock window.

    `name`/`cbid`/`correlation_id` are absent on some kinds, so collectors read them with
    `getattr`.
    """

    kind: int
    start: int
    end: int


class KernelActivity(TimedActivity, Protocol):
    """A CUPTI CONCURRENT_KERNEL record: launch shape plus the device-clock window."""

    name: str
    grid_x: int
    grid_y: int
    grid_z: int
    block_x: int
    block_y: int
    block_z: int
    static_shared_memory: int
    dynamic_shared_memory: int
    registers_per_thread: int


class MemcpyActivity(TimedActivity, Protocol):
    """A CUPTI MEMCPY record: direction code and device-clock window (`bytes` via getattr)."""

    copy_kind: int


class RawActivity(KernelActivity, MemcpyActivity, Protocol):
    """The opaque CUPTI record before kind dispatch, statically exposing every field.

    Only the subset valid for its runtime `kind` is meaningful; the superset lets a record
    reach its kind-specific reader without a cast.
    """


class DeviceProcess(Protocol):
    """One process's memory footprint on a device, from a snapshot."""

    @property
    def pid(self) -> int: ...

    @property
    def used_bytes(self) -> int: ...


class DeviceUtilization(Protocol):
    """Device compute and memory-controller utilization, in percent (0-100)."""

    @property
    def gpu_pct(self) -> int: ...

    @property
    def memory_pct(self) -> int: ...


class DeviceEnergy(Protocol):
    """Device instantaneous power draw."""

    @property
    def power_w(self) -> float: ...


class DeviceThermal(Protocol):
    """Device thermal state."""

    @property
    def temperature_c(self) -> int: ...

    @property
    def is_throttling(self) -> bool: ...

    @property
    def throttle_names(self) -> Sequence[str]: ...


class DeviceMemory(Protocol):
    """Device memory capacity and current pressure."""

    @property
    def percent_used(self) -> float: ...

    @property
    def total_gb(self) -> float: ...


class BusyDevice(Protocol):
    """The live readings needed only for contention gating."""

    @property
    def memory(self) -> DeviceMemory: ...

    @property
    def utilization(self) -> DeviceUtilization: ...


class DeviceSnapshot(Protocol):
    """One point-in-time reading of a device's sensors, as a probe backend reports it."""

    @property
    def unit_name(self) -> str: ...

    @property
    def energy(self) -> DeviceEnergy: ...

    @property
    def processes(self) -> Sequence[DeviceProcess]: ...

    @property
    def thermal(self) -> DeviceThermal: ...

    @property
    def utilization(self) -> DeviceUtilization: ...


class DeviceProbe(Protocol):
    """The device-sampling surface profiling needs, independent of the probe backend.

    vendor: `nvidia`, `amd`, `apple`, ..., matched against `Tracer.vendor`.
    label: human-readable device name, used when no reading was ever taken.
    arch_key: stable per-architecture key (`sm_90`, ...) for `arch_config`.
    peak_bandwidth_gbs: theoretical peak memory bandwidth, 0 when unknown.
    """

    @property
    def vendor(self) -> str: ...

    @property
    def label(self) -> str: ...

    @property
    def arch_key(self) -> str: ...

    @property
    def peak_bandwidth_gbs(self) -> float: ...

    @property
    def memory(self) -> DeviceMemory: ...

    @property
    def utilization(self) -> DeviceUtilization: ...

    def snapshot(self, name: str = "") -> DeviceSnapshot:
        """Point-in-time reading of this device's sensors, tagged with region `name`."""
