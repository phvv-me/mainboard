# NVIDIA annotation + deep trace: NVTX ranges and the CUPTI Activity collector.

import threading
from collections import defaultdict, deque
from contextlib import ExitStack, suppress
from dataclasses import dataclass
from importlib import import_module
from typing import TYPE_CHECKING, ClassVar, cast

from ...trace import (
    Activity,
    ActivityRecord,
    CallbackSession,
    KernelTrace,
    MemcpyTrace,
    TraceCollector,
)
from ...tracer import Marker, Tracer, Vendor

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ...protocols import RawActivity
    from .protocols import CallbackData, CudaRuntime, Cupti, Nvtx, Subscriber

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
_MEMCPY_NAME = {
    0: "unknown",
    1: "HtoD",
    2: "DtoH",
    3: "HtoA",
    4: "AtoH",
    5: "AtoA",
    6: "AtoD",
    7: "DtoA",
    8: "DtoD",
    9: "HtoH",
    10: "PtoP",
}

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


def _sync() -> None:
    """Synchronize the device so all kernels land in the CUPTI buffer before a flush."""
    if runtime := _runtime():
        runtime.cudaDeviceSynchronize()


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
                    )
                )
            elif kind == _MEMCPY:
                target.append(
                    RawMemcpy(
                        copy_kind=int(act.copy_kind),
                        start_ns=act.start,
                        end_ns=act.end,
                        bytes_moved=getattr(act, "bytes", 0),
                        correlation_id=getattr(act, "correlation_id", 0),
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


_supported_kinds: Activity | None = None


def _disable(kinds: Sequence[int]) -> None:
    """Disable exactly the native activity kinds enabled for one capture."""
    api = _cupti()
    for kind in kinds:
        api.activity_disable(kind)


@dataclass(frozen=True, slots=True)
class RawKernel:
    """Fields copied from one kernel activity before CUPTI releases its buffer."""

    name: str
    start_ns: int
    end_ns: int
    grid: str
    block: str
    static_shared_mem: int
    dynamic_shared_mem: int
    registers: int
    correlation_id: int


@dataclass(frozen=True, slots=True)
class RawMemcpy:
    """Fields copied from one memory transfer before CUPTI releases its buffer."""

    copy_kind: int
    start_ns: int
    end_ns: int
    bytes_moved: int
    correlation_id: int


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
        self.enabled_kinds: tuple[int, ...] = ()
        self.running = False

    def __enter__(self) -> CuptiCollector:
        CuptiCollector._ensure_registered()
        if _active:
            raise RuntimeError("nested CUPTI collection unsupported (single-subscriber)")
        self.enabled_kinds = CuptiCollector._enable(self.kinds)
        with ExitStack() as undo:
            undo.callback(self._abandon)
            _sync()
            _cupti().activity_flush_all(1)  # drain prior records before capture starts
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
        with self.lock:
            records = tuple(record for record in self.records if isinstance(record, RawGeneric))
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
        self.records.append(record)

    def dropped(self) -> int:
        """Return raw activity records overwritten by the bounded deque."""
        return self.dropped_records

    def flush(self) -> None:
        _sync()
        _cupti().activity_flush_all(1)

    def _abandon(self) -> None:
        """Roll the enables back after a start that never reached capture."""
        _disable(self.enabled_kinds)
        self.enabled_kinds = ()

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
    def _enable(kinds: Activity) -> tuple[int, ...]:
        """Enable requested CUPTI kinds and return the exact native members enabled.

        ``kinds`` is already reconciled against :func:`_supported`, so every kind here is
        known to enable; an error would be a real bug, not an unsupported device.
        """
        api = _cupti()
        enabled = []
        for flag, enum_name in _CUPTI_KIND.items():
            if flag not in kinds:
                continue
            kind = getattr(api.ActivityKind, enum_name)
            api.activity_enable(kind)
            _label[int(kind)] = flag.label
            enabled.append(kind)
        return tuple(enabled)

    def kernels(self) -> list[KernelTrace]:
        with self.lock:
            records = tuple(record for record in self.records if isinstance(record, RawKernel))
        return [
            KernelTrace(
                name=record.name,
                start_ns=record.start_ns,
                end_ns=record.end_ns,
                grid=record.grid,
                block=record.block,
                static_shared_mem=record.static_shared_mem,
                dynamic_shared_mem=record.dynamic_shared_mem,
                registers=record.registers,
                correlation_id=record.correlation_id,
            )
            for record in records
        ]

    def memcpys(self) -> list[MemcpyTrace]:
        with self.lock:
            records = tuple(record for record in self.records if isinstance(record, RawMemcpy))
        return [
            MemcpyTrace(
                kind=_MEMCPY_NAME.get(record.copy_kind, f"kind_{record.copy_kind}"),
                start_ns=record.start_ns,
                end_ns=record.end_ns,
                bytes_moved=record.bytes_moved,
                correlation_id=record.correlation_id,
            )
            for record in records
        ]

    def reset(self) -> None:
        self.flush()  # drain in-flight records, then drop everything so far
        with self.lock:
            self.records.clear()
            self.dropped_records = 0

    def stop(self) -> None:
        if not self.running:
            return
        with ExitStack() as teardown:
            teardown.callback(self._settle)
            teardown.callback(self._retire)
            _sync()
            _cupti().activity_flush_all(1)

    def _retire(self) -> None:
        """Leave the active stack, tolerating a session someone else already popped."""
        if _active and _active[-1] is self:
            _active.pop()

    def _settle(self) -> None:
        """Disable what this session enabled and mark it stopped, whatever else failed."""
        with ExitStack() as rest:
            rest.callback(self._mark_stopped)
            _disable(self.enabled_kinds)

    def _mark_stopped(self) -> None:
        """Forget the enables and read as not running."""
        self.enabled_kinds = ()
        self.running = False

    # force buffered records to the completion callback


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
