# NVIDIA annotation + deep trace: NVTX ranges and the CUPTI Activity collector.

from contextlib import suppress
from typing import ClassVar

from ...enums import Activity
from ...trace import TraceCollector
from ...tracer import Marker, Tracer, Vendor
from . import bindings
from .collector import CuptiCollector
from .constants import CUPTI_KIND

_supported_kinds: Activity | None = None


class NvtxTracer(Tracer):
    """NVTX ranges, plus the CUPTI deep-trace collector."""

    vendor: ClassVar[Vendor] = Vendor.NVIDIA
    label: ClassVar[str] = "nvtx"

    @classmethod
    def is_available(cls) -> bool:
        # Either capability is enough: NVTX gives annotation, CUPTI gives deep trace.
        return bindings.has_nvtx or bindings.has_cupti

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
        api = bindings.cupti
        found = Activity(0)
        for flag, enum_name in CUPTI_KIND.items():
            kind = getattr(api.ActivityKind, enum_name)
            with suppress(NotImplementedError):
                api.activity_enable(kind)
                api.activity_disable(kind)
                found |= flag
        _supported_kinds = found
        return found

    def _open(self, kinds: Activity) -> TraceCollector:
        return CuptiCollector(kinds) if bindings.has_cupti else TraceCollector()

    def start(self, name: str) -> Marker:
        """Open an overlap-safe process range and return its exact closer."""
        if not bindings.has_nvtx:
            return super().start(name)
        range_id = bindings.nvtx.start_range(name)
        return lambda: bindings.nvtx.end_range(range_id)

    def supported(self) -> Activity:
        return NvtxTracer._supported() if bindings.has_cupti else Activity(0)

    def timestamp(self) -> int:
        return bindings.cupti.get_timestamp() if bindings.has_cupti else super().timestamp()
