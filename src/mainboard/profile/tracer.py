# Vendor code-annotation backends: named timeline ranges + instantaneous marks.

import importlib
import logging
import time
from collections.abc import Sequence
from enum import StrEnum, auto
from typing import TYPE_CHECKING, ClassVar

from patos import Registry

from .trace import Activity, CallbackSession, TraceCollector

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)
type Marker = Callable[[], None]


class Vendor(StrEnum):
    """The vendors `providers/` ships a native annotation backend for.

    `DeviceProbe.vendor` is a plain string, so it compares equal to these members by value.
    """

    NVIDIA = auto()
    AMD = auto()
    APPLE = auto()
    UNKNOWN = auto()


class Tracer(Registry):
    """No-op annotation backend and the registry root for vendor tracers.

    label: short identifier for reports.
    """

    vendor: ClassVar[Vendor] = Vendor.UNKNOWN
    label: ClassVar[str] = "none"

    @classmethod
    def detect(cls, *, present: frozenset[str] = frozenset()) -> Tracer:
        """The best available tracer: one matching a vendor in `present`, else any, else no-op.

        present: `DeviceProbe.vendor` values of the GPUs on this host, as the caller found them.
        """
        importlib.import_module("mainboard.profile.providers")
        backends = [b for b in cls.implementations() if b.is_available()]
        for backend in backends:
            if backend.vendor in present:
                return backend()
        return backends[0]() if backends else cls()

    @classmethod
    def is_available(cls) -> bool:
        """Whether this backend's annotation library can be imported here."""
        return False

    def callbacks(self, domains: Sequence[str] = ("runtime", "driver")) -> CallbackSession:
        """A synchronous API-call callback session over `runtime`/`driver`/`nvtx` domains."""
        return CallbackSession()

    def collect(self, kinds: Activity = Activity.DEFAULT) -> TraceCollector:
        """A deep per-op trace collector for `kinds`, as `resolve` reconciles them."""
        return self.open(self.resolve(kinds))

    def mark(self, name: str) -> None:
        """Emit an instantaneous named event."""

    def open(self, kinds: Activity) -> TraceCollector:
        """Build the collector for already-resolved `kinds`."""
        return TraceCollector()

    def pop(self) -> None:
        """Close the most recently opened range."""

    def push(self, name: str) -> None:
        """Open a named range on the native timeline."""

    def resolve(self, kinds: Activity) -> Activity:
        """Reconcile `kinds` with `supported`, raising ValueError when there is no collector.

        `ALL` means everything this device offers, so it adapts down and logs the dropped kinds;
        an explicit kind the device cannot collect fails fast instead of silently going missing.
        """
        supported = self.supported()
        if not supported:
            raise ValueError(f"trace backend {self.label!r} has no activity collector available")
        if kinds is Activity.ALL:
            if dropped := kinds & ~supported:
                logger.info(
                    "trace: %s unavailable on this device; collecting %s",
                    dropped,
                    kinds & supported,
                )
            return kinds & supported
        if missing := kinds & ~supported:
            raise ValueError(
                f"trace kinds {missing} not supported on this device; available here: {supported}"
            )
        return kinds

    def start(self, name: str) -> Marker:
        """Open a native range and return the exact operation that closes it."""
        self.push(name)
        return self.pop

    def supported(self) -> Activity:
        """The kinds this backend can collect here; none in the base, so `collect` refuses.

        Support is device- and driver-specific (consumer GPUs lack some CUPTI kinds), so a
        backend probes the hardware.
        """
        return Activity(0)

    def timestamp(self) -> int:
        """Device-clock nanoseconds for region binning; the host clock in the base."""
        return time.perf_counter_ns()
