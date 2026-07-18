from collections.abc import Callable
from types import TracebackType
from typing import Protocol, runtime_checkable

# JSON values accepted by the Chrome/Perfetto trace-event writer.
type Json = str | int | float | bool | None | list["Json"] | dict[str, "Json"]
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


class DType(Protocol):
    """An opaque tensor data type used only as a library argument."""


class Tensor(Protocol):
    """The small tensor surface used by the storage bandwidth probe."""

    def copy_(self, source: Tensor) -> Tensor: ...


class Cuda(Protocol):
    """CUDA availability and synchronization used by the storage probe."""

    def is_available(self) -> bool: ...
    def synchronize(self) -> None: ...


class Torch(Protocol):
    """The optional Torch operations used by the storage probe."""

    cuda: Cuda
    uint8: DType

    def empty(self, size: int, *, dtype: DType, device: str) -> Tensor: ...
    def from_file(self, path: str, *, dtype: DType, size: int) -> Tensor: ...


class CompatValue(Protocol):
    """An opaque kvikio compatibility enum member."""


class CompatModeApi(Protocol):
    """The kvikio compatibility enum surface used by the probe."""

    OFF: CompatValue


@runtime_checkable
class CurrentKvikioDefaults(Protocol):
    """The current kvikio compatibility getter."""

    def is_compat_mode_preferred(self) -> bool: ...


class LegacyKvikioDefaults(Protocol):
    """The legacy kvikio compatibility getter."""

    def compat_mode(self) -> CompatValue: ...


type KvikioDefaults = CurrentKvikioDefaults | LegacyKvikioDefaults


class CuFile(Protocol):
    """A direct-storage file handle that reads into a device tensor."""

    def __enter__(self) -> CuFile: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None: ...
    def read(self, buffer: Tensor) -> int: ...


class KvikioCompatibility(Protocol):
    """The kvikio members used to reject host-bounce compatibility mode."""

    @property
    def CompatMode(self) -> CompatModeApi: ...

    @property
    def defaults(self) -> KvikioDefaults: ...


class Kvikio(KvikioCompatibility, Protocol):
    """The optional kvikio module surface used by the storage probe."""

    CuFile: Callable[[str, str], CuFile]
