# NVIDIA annotation + deep trace: NVTX ranges and the CUPTI Activity collector.

import sys
import threading
from collections import defaultdict, deque
from contextlib import ExitStack, suppress
from ctypes import addressof, c_size_t
from dataclasses import asdict, dataclass
from importlib import import_module
from itertools import islice
from typing import TYPE_CHECKING, ClassVar, cast

from ...trace import (
    Activity,
    ActivityRecord,
    CallbackSession,
    KernelTrace,
    MemcpyTrace,
    TraceCollector,
    memcpy_kind,
)
from ...tracer import Marker, Tracer, Vendor

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ...protocols import RawActivity
    from .protocols import CallbackData, CudaDriver, CudaRuntime, Cupti, Nvtx, Subscriber

# The accelerator bindings ship no stubs, so each handle is `import_module`'s untyped
# `ModuleType`, cast to its `protocols.py` Protocol.
nvtx: Nvtx | None = None
with suppress(ImportError):
    nvtx = cast("Nvtx", import_module("nvtx"))

cupti: Cupti | None = None
with suppress(ImportError):
    cupti = cast("Cupti", import_module("cupti.cupti"))  # the `cupti` package's `cupti` submodule

cuda_runtime: CudaRuntime | None = None
_runtime_loaded = False

_CONCURRENT_KERNEL = 10  # int(cupti.ActivityKind.CONCURRENT_KERNEL) — literal per CUPTI rule
_MEMCPY = 1  # int(cupti.ActivityKind.MEMCPY)
_BUFFER_SIZE = 8 * 1024 * 1024
_MAX_RECORDS = 262_144

# Each Activity flag -> its CUPTI ActivityKind enum-member name.
_CUPTI_KIND = {
    Activity.KERNEL: "CONCURRENT_KERNEL",
    Activity.MEMCPY: "MEMCPY",
    Activity.MEMSET: "MEMSET",
    Activity.SYNC: "SYNCHRONIZATION",
    Activity.OVERHEAD: "OVERHEAD",
    Activity.MEMORY: "MEMORY",
    Activity.JIT: "JIT",
    Activity.RUNTIME: "RUNTIME",
    Activity.DRIVER: "DRIVER",
    Activity.MEMORY_POOL: "MEMORY_POOL",
}

_active: list[CuptiCollector] = []
_registered = False
_label: dict[int, str] = {}  # activity-kind int -> friendly label (built as kinds enable)
_domain: dict[int, int] = {}  # kind int -> CallbackDomain, for cbid -> function-name lookup
_supported_kinds: Activity | None = None


def _sync() -> None:
    """Synchronize the device so all kernels land in the CUPTI buffer before a flush."""
    runtime = _runtime()
    if runtime is None:
        raise RuntimeError("CUDA runtime binding is required to synchronize activity capture")
    (status,) = runtime.cudaDeviceSynchronize()
    if status != 0:  # cudaSuccess is zero; asynchronous execution errors surface here too.
        raise RuntimeError(f"cudaDeviceSynchronize failed with CUDA status {status}")


def _runtime() -> CudaRuntime | None:
    """Load CUDA Runtime only when a deep trace actually needs its synchronization barrier."""
    global cuda_runtime, _runtime_loaded
    if not _runtime_loaded:
        _runtime_loaded = True
        with suppress(ImportError, OSError):
            cuda_runtime = cast("CudaRuntime", import_module("cuda.bindings.runtime"))
    return cuda_runtime


def _cupti() -> Cupti:
    """The loaded CUPTI module — every caller reaches here only when CUPTI is present."""
    assert cupti is not None  # noqa: S101  reason=type narrowing, guaranteed by is_available() since=2026-08-16
    return cupti


def _on_buffer_requested() -> tuple[int, int]:
    return _BUFFER_SIZE, 0


def _on_buffer_completed(activities: Sequence[RawActivity]) -> None:
    """Copy bounded raw records while CUPTI still owns the activity objects."""
    if not _active:
        return
    target = _active[-1]
    with target.lock:
        complete = False
        try:
            for act in activities:
                kind = int(act.kind)
                if kind == _CONCURRENT_KERNEL:
                    target.append(
                        RawKernel(
                            name=act.name,
                            start_ns=act.start,
                            end_ns=act.end,
                            grid=f"{act.grid_x}x{act.grid_y}x{act.grid_z}",
                            block=f"{act.block_x}x{act.block_y}x{act.block_z}",
                            static_shared_mem=act.static_shared_memory,
                            dynamic_shared_mem=act.dynamic_shared_memory,
                            registers=act.registers_per_thread,
                            correlation_id=getattr(act, "correlation_id", 0),
                            device_id=getattr(act, "device_id", None),
                            context_id=getattr(act, "context_id", None),
                            stream_id=getattr(act, "stream_id", None),
                        )
                    )
                elif kind == _MEMCPY:
                    target.append(
                        RawMemcpy(
                            kind=memcpy_kind(int(act.copy_kind)),
                            start_ns=act.start,
                            end_ns=act.end,
                            bytes_moved=getattr(act, "bytes", 0),
                            correlation_id=getattr(act, "correlation_id", 0),
                            device_id=getattr(act, "device_id", None),
                            context_id=getattr(act, "context_id", None),
                            stream_id=getattr(act, "stream_id", None),
                        )
                    )
                elif kind in _label:
                    target.append(
                        RawGeneric(
                            kind_id=kind,
                            kind=_label[kind],
                            name=getattr(act, "name", None),
                            cbid=getattr(act, "cbid", None),
                            start_ns=act.start,
                            end_ns=act.end,
                            correlation_id=getattr(act, "correlation_id", 0),
                        )
                    )
            complete = True
        finally:
            target.callback_failed |= not complete


def _disable(kinds: Sequence[int]) -> None:
    """Disable exactly the native activity kinds enabled for one capture."""
    api = _cupti()
    cleanup = ExitStack()
    for kind in kinds:
        cleanup.callback(api.activity_disable, kind)
    # An inner ``with`` would hide the active work/start error from this stack.
    cleanup.__exit__(*sys.exc_info())


@dataclass(frozen=True, slots=True)
class RawKernel:
    """`KernelTrace`'s fields, copied cheaply before CUPTI releases its buffer."""

    name: str
    start_ns: int
    end_ns: int
    grid: str
    block: str
    static_shared_mem: int
    dynamic_shared_mem: int
    registers: int
    correlation_id: int
    device_id: int | None = None
    context_id: int | None = None
    stream_id: int | None = None


@dataclass(frozen=True, slots=True)
class RawMemcpy:
    """`MemcpyTrace`'s fields, copied cheaply before CUPTI releases its buffer."""

    kind: str
    start_ns: int
    end_ns: int
    bytes_moved: int
    correlation_id: int
    device_id: int | None = None
    context_id: int | None = None
    stream_id: int | None = None


@dataclass(frozen=True, slots=True)
class RawGeneric:
    """Fields copied from one generic activity before deferred name resolution."""

    kind_id: int
    kind: str
    name: str | None
    cbid: int | None
    start_ns: int
    end_ns: int
    correlation_id: int


type RawRecord = RawKernel | RawMemcpy | RawGeneric


class CuptiCollector(TraceCollector):
    """Asynchronous CUPTI Activity collector — single-subscriber, low overhead.

    kinds: the :class:`Activity` flags to collect. ``KERNEL``/``MEMCPY`` become typed
    records; the rest become generic :class:`ActivityRecord`s.
    """

    def __init__(
        self, kinds: Activity = Activity.DEFAULT, max_records: int = _MAX_RECORDS
    ) -> None:
        self.lock = threading.Lock()
        self.kinds = kinds
        self.records: deque[RawRecord] = deque(maxlen=max_records)
        self.dropped_records = 0
        self.native_dropped_records = 0
        self.enabled_kinds: tuple[int, ...] = ()
        self.running = False
        self.received_records = 0
        self.lost_records = 0
        self.callback_failed = False
        self.scope: tuple[int, int, int] | None = None
        self.owner_thread = threading.get_ident()

    def __enter__(self) -> CuptiCollector:
        CuptiCollector._ensure_registered()
        if _active:
            raise RuntimeError("nested CUPTI collection unsupported (single-subscriber)")
        self.enabled_kinds = CuptiCollector._enable(self.kinds)
        self.owner_thread = threading.get_ident()
        with ExitStack() as undo:
            undo.callback(self._settle)
            _sync()
            self.scope = self._scope()
            _cupti().activity_flush_all(1)  # drain prior records before capture starts
            CuptiCollector._native_drops()  # reset the global count from before this capture
            _active.append(self)
            self.running = True
            undo.pop_all()
        return self

    @staticmethod
    def activity_name(record: RawGeneric) -> str:
        """Resolve an API function name after the CUPTI callback has returned."""
        if record.name:
            return record.name
        domain = _domain.get(record.kind_id)
        if record.cbid is not None and domain is not None and cupti is not None:
            return cupti.get_callback_name(domain, record.cbid)
        return record.kind

    def activities(self) -> list[ActivityRecord]:
        records = (
            record for record in self._records(None, None) if isinstance(record, RawGeneric)
        )
        return [
            ActivityRecord(
                kind=record.kind,
                name=self.activity_name(record),
                start_ns=record.start_ns,
                end_ns=record.end_ns,
                correlation_id=record.correlation_id,
            )
            for record in records
        ]

    def append(self, record: RawRecord) -> None:
        """Append one raw record while keeping capture memory bounded."""
        if len(self.records) == self.records.maxlen:
            self.dropped_records += 1
            self.lost_records += 1
        self.records.append(record)
        self.received_records += 1

    def checkpoint(self, activities: Activity) -> tuple[int, int]:
        """Synchronize every stream in the captured context and freeze a raw cursor.

        Both counters are monotone across reset(); an old window cannot mistake a
        cleared or overwritten buffer for complete evidence.
        """
        if not self.running:
            raise RuntimeError("activity checkpoints require a running collector")
        enabled = Activity(0)
        for flag, name in _CUPTI_KIND.items():
            if getattr(_cupti().ActivityKind, name) in self.enabled_kinds:
                enabled |= flag
        if activities & ~enabled:
            raise RuntimeError("activity window requested kinds not actually enabled")
        self.flush()
        with self.lock:
            return self.received_records, self.lost_records

    @property
    def device_index(self) -> int:
        if self.scope is None:
            raise RuntimeError("activity capture has no current CUDA context")
        return self.scope[0]

    def dropped(self) -> int:
        """Return native buffer loss plus deque overwrites through the latest flush."""
        with self.lock:
            return self.native_dropped_records + self.dropped_records

    def flush(self) -> None:
        if self.scope is not None and (
            threading.get_ident() != self.owner_thread or self._scope() != self.scope
        ):
            raise RuntimeError("activity capture changed its issuing thread or CUDA context")
        _sync()
        _cupti().activity_flush_all(1)
        dropped = CuptiCollector._native_drops()
        with self.lock:
            self.native_dropped_records += dropped
            self.lost_records += dropped
            if self.callback_failed:
                raise RuntimeError("activity collector tainted by callback conversion failure")

    def kernels(self, *, since: int | None = None, until: int | None = None) -> list[KernelTrace]:
        records = self._records(since, until)
        return [
            KernelTrace(**asdict(record)) for record in records if isinstance(record, RawKernel)
        ]

    def memcpys(self, *, since: int | None = None, until: int | None = None) -> list[MemcpyTrace]:
        records = self._records(since, until)
        return [
            MemcpyTrace(**asdict(record)) for record in records if isinstance(record, RawMemcpy)
        ]

    def reset(self) -> None:
        self.flush()  # drain in-flight records, then drop everything so far
        with self.lock:
            self.records.clear()
            self.dropped_records = 0
            self.native_dropped_records = 0

    def stop(self) -> None:
        if not self.running:
            return
        teardown = ExitStack()
        teardown.callback(self._settle)
        teardown.callback(self._retire)
        try:
            self.flush()
        finally:
            teardown.__exit__(*sys.exc_info())

    @staticmethod
    def _enable(kinds: Activity) -> tuple[int, ...]:
        """Enable requested CUPTI kinds and return the exact native members enabled.

        ``kinds`` is already reconciled against :func:`_supported`, so every kind here is
        known to enable; an error would be a real bug, not an unsupported device.
        """
        api = _cupti()
        enabled: list[int] = []
        with ExitStack() as rollback:
            rollback.callback(_disable, enabled)
            for flag, enum_name in _CUPTI_KIND.items():
                if flag not in kinds:
                    continue
                kind = getattr(api.ActivityKind, enum_name)
                api.activity_enable(kind)
                enabled.append(kind)
                _label[int(kind)] = flag.label
            rollback.pop_all()
        return tuple(enabled)

    @staticmethod
    def _ensure_registered() -> None:
        """Register the activity callbacks once, for the process (never unregistered)."""
        global _registered
        if _registered:
            return
        api = _cupti()
        api.activity_register_callbacks(_on_buffer_requested, _on_buffer_completed)
        _domain[int(api.ActivityKind.RUNTIME)] = api.CallbackDomain.RUNTIME_API
        _domain[int(api.ActivityKind.DRIVER)] = api.CallbackDomain.DRIVER_API
        _registered = True

    @staticmethod
    def _native_drops() -> int:
        """Drain CUPTI's reset-on-read global queue loss count, including empty flushes.

        CUDA 6 onward delivers global buffers, so context and stream are both zero.
        cupti-python takes the address of a size_t output, not an integer result.
        """
        dropped = c_size_t()
        _cupti().activity_get_num_dropped_records(0, 0, addressof(dropped))
        return dropped.value

    def _mark_stopped(self) -> None:
        """Forget the enables and read as not running."""
        self.enabled_kinds = ()
        self.running = False

    def _records(self, since: int | None, until: int | None) -> tuple[RawRecord, ...]:
        """Copy only one delivered range; never re-materialize the entire outer trace."""
        with self.lock:
            if since is None:
                return tuple(self.records)
            end = self.received_records if until is None else until
            first = self.received_records - len(self.records)
            if not first <= since <= end <= self.received_records:
                raise RuntimeError("activity window was reset or overwritten before retrieval")
            records = tuple(
                islice(
                    reversed(self.records),
                    self.received_records - end,
                    self.received_records - since,
                )
            )[::-1]
        # Global CUPTI delivery is wider than one context. Do not silently filter
        # foreign or unidentifiable GPU work and then call the remainder complete.
        for record in records:
            if not isinstance(record, (RawKernel, RawMemcpy)):
                continue
            if self.scope is None or (record.device_id, record.context_id) != self.scope[1:]:
                raise RuntimeError("activity window contains a foreign or unknown CUDA context")
            if record.stream_id is None or record.start_ns <= 0 or record.end_ns < record.start_ns:
                raise RuntimeError(
                    "activity window contains incomplete native timing or stream data"
                )
        return records

    @staticmethod
    def _scope() -> tuple[int, int, int]:
        """Actual visible ordinal, CUPTI device ID, and CUPTI context ID; no substitution."""
        runtime = _runtime()
        if runtime is None:
            raise RuntimeError("CUDA runtime is required to identify an activity window")
        status, device = runtime.cudaGetDevice()
        if status != 0:
            raise RuntimeError(f"cudaGetDevice failed with CUDA status {status}")
        driver = cast("CudaDriver", import_module("cuda.bindings.driver"))
        status, context = driver.cuCtxGetCurrent()
        if status != 0 or not int(context):
            raise RuntimeError(f"cuCtxGetCurrent failed or returned no context: {status}")
        return device, _cupti().get_device_id(int(context)), _cupti().get_context_id(int(context))

    def _retire(self) -> None:
        """Leave the active stack, tolerating a session someone else already popped."""
        if _active and _active[-1] is self:
            _active.pop()

    def _settle(self) -> None:
        """Disable what this session enabled and mark it stopped, whatever else failed."""
        with ExitStack() as rest:
            rest.callback(self._mark_stopped)
            _disable(self.enabled_kinds)


_CB_DOMAIN_NAME = {"runtime": "RUNTIME_API", "driver": "DRIVER_API", "nvtx": "NVTX"}


class CuptiCallbackSession(CallbackSession):
    """CUPTI Callback API subscription — counts CUDA API calls by name, synchronously."""

    def __init__(self, domains: Sequence[str]) -> None:
        self.domains = domains
        self._subscriber: Subscriber | None = None
        self._counts: defaultdict[str, int] = defaultdict(int)
        self._lock = threading.Lock()

    def __enter__(self) -> CuptiCallbackSession:
        api = _cupti()
        self._subscriber = api.subscribe(self._on_callback, None)
        for name in self.domains:
            domain = getattr(api.CallbackDomain, _CB_DOMAIN_NAME.get(name, ""), None)
            if domain is not None:
                api.enable_domain(1, self._subscriber, domain)
        return self

    def counts(self) -> dict[str, int]:
        with self._lock:
            return dict(self._counts)

    def stop(self) -> None:
        if self._subscriber is not None:
            _cupti().unsubscribe(self._subscriber)
            self._subscriber = None

    def _on_callback(self, _userdata: None, _domain: int, _cbid: int, data: CallbackData) -> None:
        if data.callback_site == _cupti().ApiCallbackSite.API_ENTER:
            with self._lock:
                self._counts[data.function_name] += 1


class NvtxTracer(Tracer):
    """NVTX push/pop ranges and marks, plus the CUPTI deep-trace collector."""

    vendor: ClassVar[Vendor] = Vendor.NVIDIA
    label: ClassVar[str] = "nvtx"

    @classmethod
    def is_available(cls) -> bool:
        # Either capability is enough: NVTX gives annotation, CUPTI gives deep trace.
        return nvtx is not None or cupti is not None

    def callbacks(self, domains: Sequence[str] = ("runtime", "driver")) -> CallbackSession:
        return CuptiCallbackSession(domains) if cupti is not None else CallbackSession()

    def mark(self, name: str) -> None:
        if nvtx is not None:
            nvtx.mark(message=name)

    def open(self, kinds: Activity) -> TraceCollector:
        return CuptiCollector(kinds) if cupti is not None else TraceCollector()

    def pop(self) -> None:
        if nvtx is not None:
            nvtx.pop_range()

    def push(self, name: str) -> None:
        if nvtx is not None:
            nvtx.push_range(name)

    def start(self, name: str) -> Marker:
        """Open an overlap-safe process range and return its exact closer."""
        api = nvtx
        if api is None:
            return super().start(name)
        range_id = api.start_range(name)
        return lambda: api.end_range(range_id)

    def supported(self) -> Activity:
        return NvtxTracer._supported() if cupti is not None else Activity(0)

    def timestamp(self) -> int:
        return int(cupti.get_timestamp()) if cupti is not None else super().timestamp()

    @staticmethod
    def _supported() -> Activity:
        """The Activity kinds CUPTI can enable on this device, probed once and cached.

        ``activity_enable`` is the capability gate — it raises ``NotImplementedError`` for a
        kind the device/driver does not implement (e.g. ``MEMORY`` on GB10, or PC sampling
        anywhere in cupti-python). We toggle each candidate on then straight off and keep the
        set that took. Probing runs before any collector registers buffer callbacks, so the
        transient enable/disable cannot drop real records.
        """
        global _supported_kinds
        if _supported_kinds is not None:
            return _supported_kinds
        api = _cupti()
        found = Activity(0)
        for flag, enum_name in _CUPTI_KIND.items():
            kind = getattr(api.ActivityKind, enum_name)
            try:
                api.activity_enable(kind)
            except NotImplementedError:
                continue
            api.activity_disable(kind)
            found |= flag
        _supported_kinds = found
        return found
