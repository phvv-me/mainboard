# NVIDIA annotation + deep trace: NVTX ranges, CUPTI callbacks, and the Activity collector.

import threading
from collections import defaultdict
from contextlib import suppress
from typing import TYPE_CHECKING, ClassVar

from ...trace import Activity, CallbackSession, TraceCollector
from ...tracer import Marker, Tracer, Vendor
from .bindings import cupti, loaded, nvtx
from .collector import CuptiCollector
from .constants import CUPTI_KIND

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .bindings import CallbackData, Subscriber

_supported_kinds: Activity | None = None


_CB_DOMAIN_NAME = {"runtime": "RUNTIME_API", "driver": "DRIVER_API", "nvtx": "NVTX"}


class CuptiCallbackSession(CallbackSession):
    """CUPTI Callback API subscription — counts CUDA API calls by name, synchronously."""

    _counts: defaultdict[str, int]
    _lock: threading.Lock
    _subscriber: Subscriber | None
    domains: Sequence[str]

    def __enter__(self) -> CuptiCallbackSession:
        api = loaded()
        self._subscriber = api.subscribe(self._on_callback, None)
        for name in self.domains:
            domain = getattr(api.CallbackDomain, _CB_DOMAIN_NAME.get(name, ""), None)
            if domain is not None:
                api.enable_domain(1, self._subscriber, domain)
        return self

    def __init__(self, domains: Sequence[str]) -> None:
        self.domains = domains
        self._subscriber = None
        self._counts = defaultdict(int)
        self._lock = threading.Lock()

    def _on_callback(self, _userdata: None, _domain: int, _cbid: int, data: CallbackData) -> None:
        if data.callback_site == loaded().ApiCallbackSite.API_ENTER:
            with self._lock:
                self._counts[data.function_name] += 1

    def counts(self) -> dict[str, int]:
        with self._lock:
            return dict(self._counts)

    def stop(self) -> None:
        if self._subscriber is not None:
            loaded().unsubscribe(self._subscriber)
            self._subscriber = None


class NvtxTracer(Tracer):
    """NVTX push/pop ranges and marks, plus the CUPTI deep-trace collector."""

    vendor: ClassVar[Vendor] = Vendor.NVIDIA
    label: ClassVar[str] = "nvtx"

    @classmethod
    def is_available(cls) -> bool:
        # Either capability is enough: NVTX gives annotation, CUPTI gives deep trace.
        return nvtx is not None or cupti is not None

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
        api = loaded()
        found = Activity(0)
        for flag, enum_name in CUPTI_KIND.items():
            kind = getattr(api.ActivityKind, enum_name)
            with suppress(NotImplementedError):
                api.activity_enable(kind)
                api.activity_disable(kind)
                found |= flag
        _supported_kinds = found
        return found

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
        return cupti.get_timestamp() if cupti is not None else super().timestamp()
