# Structural contracts profiling reads through, so it never names a vendor backend. The
# `Profiler` reaches the probe package's own vendor-neutral registry to discover a host's
# devices; nothing here, and nothing that reads a finished `Profile`, knows a backend exists.

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Sequence

# JSON values accepted by the Chrome/Perfetto trace-event writer.
type Json = str | int | float | bool | list[Json] | dict[str, Json] | None
type TraceEvent = dict[str, Json]


class TimedActivity(Protocol):
    """Any timed CUPTI record: its kind and device-clock window.

    CUPTI buffers yield one opaque record family discriminated at runtime by `kind`; this
    is the field set every record carries. The kind-specific fields below extend it, and
    the collector dispatches on `kind` before reading them. `name`/`cbid`/`correlation_id`
    are absent on some kinds, so the collector still reads those defensively with `getattr`.
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
    """The opaque CUPTI record as the buffer hands it over, before kind dispatch.

    CUPTI yields one C struct family, so a single record statically exposes every field;
    only the subset valid for its runtime `kind` is meaningful. Typing the buffer as this
    superset lets the collector pass a record to the kind-specific reader without a cast,
    and the reader takes only the fields its kind defines.
    """


class DeviceProcess(Protocol):
    """One process's memory footprint on a device, from a snapshot."""

    pid: int
    used_bytes: int


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

    temperature_c: int

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

    unit_name: str

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

    vendor: hardware vendor string (`nvidia`, `amd`, `apple`, ...), matched against a
        `Tracer`'s own `vendor` to pick the native annotation backend.
    label: human-readable device name, used when no reading was ever taken.
    arch_key: stable per-architecture dispatch key (`sm_90`, ...), for `arch_config`.
    peak_bandwidth_gbs: theoretical peak memory bandwidth, 0 when unknown.
    """

    vendor: str
    label: str
    arch_key: str
    peak_bandwidth_gbs: float

    @property
    def memory(self) -> DeviceMemory: ...

    @property
    def utilization(self) -> DeviceUtilization: ...

    def snapshot(self, name: str = "") -> DeviceSnapshot:
        """Point-in-time reading of this device's sensors, tagged with region `name`."""
