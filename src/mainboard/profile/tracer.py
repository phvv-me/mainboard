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
    """The hardware vendors a native annotation backend can match against.

    Scoped to what `providers/` ships a tracer for. `DeviceProbe.vendor` (the seam a
    caller reads off `mainboard.probe`) is a plain string, so any vendor value there
    compares equal to these members by string value regardless of which enum minted it.
    """

    NVIDIA = auto()
    AMD = auto()
    APPLE = auto()
    UNKNOWN = auto()


class Tracer(Registry):
    """No-op annotation backend and the registry root for vendor tracers.

    vendor: the hardware vendor this backend annotates for.
    label: short identifier for reports.
    """

    vendor: ClassVar[Vendor] = Vendor.UNKNOWN
    label: ClassVar[str] = "none"

    @classmethod
    def detect(cls, *, present: frozenset[str] = frozenset()) -> Tracer:
        """The best available tracer: one matching a vendor in `present`, else any, else no-op.

        present: vendors of GPUs actually on this host (`DeviceProbe.vendor` values), so
            the caller decides what is present rather than this module probing for it.
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
        """A synchronous API-call callback session (no-op base; vendor backends override).

        domains: which callback domains to subscribe to (``runtime``/``driver``/``nvtx``).
        """
        return CallbackSession()

    def collect(self, kinds: Activity = Activity.DEFAULT) -> TraceCollector:
        """A deep per-op trace collector for ``kinds``, resolved against device support.

        ``Activity.ALL`` means "everything this device offers", so it *adapts* down to
        the supported subset (dropped kinds are logged). Any *explicitly* requested kind
        the device cannot collect *fails fast* with :class:`ValueError` — better a clear
        error than a profile that silently omits what you asked for.
        """
        return self.open(self.resolve(kinds))

    def mark(self, name: str) -> None:
        """Emit an instantaneous named event."""

    def open(self, kinds: Activity) -> TraceCollector:
        """Build the collector for already-resolved ``kinds`` (no-op base; backends override)."""
        return TraceCollector()

    def pop(self) -> None:
        """Close the most recently opened range."""

    def push(self, name: str) -> None:
        """Open a named range on the native timeline (no-op in the base)."""

    def resolve(self, kinds: Activity) -> Activity:
        """Reconcile requested ``kinds`` with :meth:`supported`: adapt ALL, else fail fast."""
        supported = self.supported()
        if not supported:  # backend reports no support (e.g. no-op base) -> don't second-guess
            return kinds
        if kinds is Activity.ALL:
            dropped = kinds & ~supported
            if dropped:
                logger.info(
                    "trace: %s unavailable on this device; collecting %s",
                    dropped,
                    kinds & supported,
                )
            return kinds & supported
        missing = kinds & ~supported
        if missing:
            raise ValueError(
                f"trace kinds {missing} not supported on this device; available here: {supported}"
            )
        return kinds

    def start(self, name: str) -> Marker:
        """Open a native range and return the exact operation that closes it."""
        self.push(name)
        return self.pop

    def supported(self) -> Activity:
        """The :class:`Activity` kinds this backend can collect on the current device.

        Support is device- and driver-specific (e.g. consumer GPUs lack some CUPTI
        kinds), so a backend probes the hardware. The base supports none — it has no
        deep trace — which makes :meth:`collect` a silent no-op rather than an error
        on a host with no profiling backend.
        """
        return Activity(0)

    def timestamp(self) -> int:
        """Device-clock timestamp (ns) for region binning; host clock in the base."""
        return time.perf_counter_ns()
