"""NVIDIA annotation + deep trace: NVTX ranges and the CUPTI Activity collector.

Deep tracing uses the CUPTI **Activity** API only (asynchronous, buffered kernel +
memcpy records) — never counter/PC-sampling/replay — so it adds no work to the
application's launch path. Callbacks are registered once and never disabled (CUPTI
drops activity from regions where callbacks are re-registered after a disable); a
module-level active stack routes records to the live collector. A single device sync
drains buffers at session boundaries; records carry GPU-clock timestamps so the
profiler bins them into regions afterwards with no per-region synchronize.
"""

import threading
from collections import defaultdict
from contextlib import suppress
from importlib import import_module
from typing import TYPE_CHECKING, ClassVar

from ...enums import Vendor
from ...profiling.trace import (
    Activity,
    ActivityRecord,
    CallbackSession,
    KernelTrace,
    MemcpyTrace,
    TraceCollector,
)
from ...profiling.tracer import Tracer

if TYPE_CHECKING:
    from ...profiling.protocols import RawActivity
    from .protocols import CallbackData, Cupti, Nvtx, Subscriber
    from .protocols import CudaRuntime as CudaRuntimeApi


# The accelerator bindings ship no type stubs, so the handles are declared as the Protocols
# in `protocols.py` (the surface mainboard calls) and bound to the real modules at runtime;
# each is `None` when its package is absent. `import_module` returns an untyped `ModuleType`
# that pyrefly cannot bridge to a Protocol (its `__getattr__` defeats the structural check),
# so the one assignment per loader carries a `pyrefly: ignore` for that genuine stub gap.
def _load_nvtx() -> Nvtx:
    return import_module("nvtx")  # pyrefly: ignore[bad-return]


def _load_cupti() -> Cupti:  # the `cupti` package's `cupti` submodule
    return import_module("cupti.cupti")  # pyrefly: ignore[bad-return]


def _load_runtime() -> CudaRuntimeApi:
    return import_module("cuda.bindings.runtime")  # pyrefly: ignore[bad-return]


nvtx: Nvtx | None = None
with suppress(ImportError):
    nvtx = _load_nvtx()

cupti: Cupti | None = None
with suppress(ImportError):
    cupti = _load_cupti()

cuda_runtime: CudaRuntimeApi | None = None
with suppress(ImportError):
    cuda_runtime = _load_runtime()

_CONCURRENT_KERNEL = 10  # int(cupti.ActivityKind.CONCURRENT_KERNEL) — literal per CUPTI rule
_MEMCPY = 1  # int(cupti.ActivityKind.MEMCPY)
_BUFFER_SIZE = 8 * 1024 * 1024

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
    if cuda_runtime is not None:
        cuda_runtime.cudaDeviceSynchronize()


def _cupti() -> Cupti:
    """The loaded CUPTI module — every caller reaches here only when CUPTI is present."""
    assert cupti is not None
    return cupti


def _on_buffer_requested() -> tuple[int, int]:
    return _BUFFER_SIZE, 0


def _record_name(act: RawActivity, label: str) -> str:
    """Resolve a record's name: kernel name, else API function (via cbid), else the kind."""
    name: str | None = getattr(act, "name", None)
    if name:
        return name
    cbid: int | None = getattr(act, "cbid", None)
    domain = _domain.get(int(act.kind))
    if cbid is not None and domain is not None and cupti is not None:
        return cupti.get_callback_name(domain, cbid)
    return label


def _on_buffer_completed(activities: list[RawActivity]) -> None:
    """Route a completed buffer to the live collector — runs off the launch path.

    Parsing here is necessary (the raw activity is only valid during the callback) but
    happens on a CUPTI worker, not the application's compute stream.
    """
    if not _active:
        return
    target = _active[-1]
    kernels, memcpys, others = [], [], []
    for act in activities:
        kind = int(act.kind)
        if kind == _CONCURRENT_KERNEL:
            kernels.append(KernelTrace.from_activity(act))
        elif kind == _MEMCPY:
            memcpys.append(MemcpyTrace.from_activity(act))
        elif kind in _label:
            others.append(
                ActivityRecord(
                    kind=_label[kind],
                    name=_record_name(act, _label[kind]),
                    start_ns=act.start,
                    end_ns=act.end,
                    correlation_id=getattr(act, "correlation_id", 0),
                )
            )
    with target.lock:
        target.kernel_records.extend(kernels)
        target.memcpy_records.extend(memcpys)
        target.activity_records.extend(others)


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


_supported_kinds: Activity | None = None


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


def _enable(kinds: Activity) -> None:
    """Enable each requested Activity flag's CUPTI kind (idempotent; never disabled).

    ``kinds`` is already reconciled against :func:`_supported`, so every kind here is
    known to enable; an error would be a real bug, not an unsupported device.
    """
    api = _cupti()
    for flag, enum_name in _CUPTI_KIND.items():
        if flag not in kinds:
            continue
        kind = getattr(api.ActivityKind, enum_name)
        api.activity_enable(kind)
        _label[int(kind)] = flag.label


class CuptiCollector(TraceCollector):
    """Asynchronous CUPTI Activity collector — single-subscriber, low overhead.

    kinds: the :class:`Activity` flags to collect. ``KERNEL``/``MEMCPY`` become typed
    records; the rest become generic :class:`ActivityRecord`s.
    """

    def __init__(self, kinds: Activity = Activity.DEFAULT) -> None:
        self.lock = threading.Lock()
        self.kinds = kinds
        self.kernel_records: list[KernelTrace] = []
        self.memcpy_records: list[MemcpyTrace] = []
        self.activity_records: list[ActivityRecord] = []

    def __enter__(self) -> CuptiCollector:
        _ensure_registered()
        if _active:
            raise RuntimeError("nested CUPTI collection unsupported (single-subscriber)")
        _enable(self.kinds)
        _sync()
        _cupti().activity_flush_all(1)  # drain any prior region's records before we start
        _active.append(self)
        return self

    def stop(self) -> None:
        _sync()
        _cupti().activity_flush_all(1)
        if _active and _active[-1] is self:
            _active.pop()

    def reset(self) -> None:
        self.flush()  # drain in-flight records, then drop everything so far
        with self.lock:
            self.kernel_records.clear()
            self.memcpy_records.clear()
            self.activity_records.clear()

    def flush(self) -> None:
        _sync()
        _cupti().activity_flush_all(1)  # force buffered records to the completion callback

    def kernels(self) -> list[KernelTrace]:
        with self.lock:
            return list(self.kernel_records)

    def memcpys(self) -> list[MemcpyTrace]:
        with self.lock:
            return list(self.memcpy_records)

    def activities(self) -> list[ActivityRecord]:
        with self.lock:
            return list(self.activity_records)


_CB_DOMAIN_NAME = {"runtime": "RUNTIME_API", "driver": "DRIVER_API", "nvtx": "NVTX"}


class CuptiCallbackSession(CallbackSession):
    """CUPTI Callback API subscription — counts CUDA API calls by name, synchronously."""

    def __init__(self, domains: tuple[str, ...]) -> None:
        self.domains = domains
        self._subscriber: Subscriber | None = None
        self._counts: defaultdict[str, int] = defaultdict(int)
        self._lock = threading.Lock()

    def _on_callback(self, _userdata: None, _domain: int, _cbid: int, data: CallbackData) -> None:
        if data.callback_site == _cupti().ApiCallbackSite.API_ENTER:
            with self._lock:
                self._counts[data.function_name] += 1

    def __enter__(self) -> CuptiCallbackSession:
        api = _cupti()
        self._subscriber = api.subscribe(self._on_callback, None)
        for name in self.domains:
            domain = getattr(api.CallbackDomain, _CB_DOMAIN_NAME.get(name, ""), None)
            if domain is not None:
                api.enable_domain(1, self._subscriber, domain)
        return self

    def stop(self) -> None:
        if self._subscriber is not None:
            _cupti().unsubscribe(self._subscriber)
            self._subscriber = None

    def counts(self) -> dict[str, int]:
        with self._lock:
            return dict(self._counts)


class NvtxTracer(Tracer):
    """NVTX push/pop ranges and marks, plus the CUPTI deep-trace collector."""

    vendor: ClassVar[Vendor] = Vendor.NVIDIA
    label: ClassVar[str] = "nvtx"

    @classmethod
    def is_available(cls) -> bool:
        # Either capability is enough: NVTX gives annotation, CUPTI gives deep trace.
        return nvtx is not None or cupti is not None

    def push(self, name: str) -> None:
        if nvtx is not None:
            nvtx.push_range(name)

    def pop(self) -> None:
        if nvtx is not None:
            nvtx.pop_range()

    def mark(self, name: str) -> None:
        if nvtx is not None:
            nvtx.mark(message=name)

    def supported(self) -> Activity:
        return _supported() if cupti is not None else Activity(0)

    def open(self, kinds: Activity) -> TraceCollector:
        return CuptiCollector(kinds) if cupti is not None else TraceCollector()

    def callbacks(self, domains: tuple[str, ...] = ("runtime", "driver")) -> CallbackSession:
        return CuptiCallbackSession(domains) if cupti is not None else CallbackSession()

    def timestamp(self) -> int:
        return int(cupti.get_timestamp()) if cupti is not None else super().timestamp()
