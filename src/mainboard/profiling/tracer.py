"""Vendor code-annotation backends: named timeline ranges + instantaneous marks.

A `Tracer` emits the vendor's native annotation (NVTX / ROCTx / `os_signpost`) so a
region shows up on the native timeline (Nsight / `rocprofv3` / Instruments) and is
near-free when no external profiler is attached. The base is a no-op, so annotation
compiles away on a host with no backend. Concrete vendor tracers live next to their
GPU provider and self-register via :class:`Registry`; :meth:`detect` picks the one
whose library is importable, preferring a match for a GPU actually present.
"""

import logging
import time
from typing import ClassVar

from patos import Registry

from ..enums import Vendor
from ..gpu import GPU
from .trace import Activity, CallbackSession, TraceCollector

logger = logging.getLogger(__name__)


class Tracer(Registry):
    """No-op annotation backend and the registry root for vendor tracers.

    vendor: the hardware vendor this backend annotates for.
    label: short identifier for reports.
    """

    vendor: ClassVar[Vendor] = Vendor.UNKNOWN
    label: ClassVar[str] = "none"

    @classmethod
    def is_available(cls) -> bool:
        """Whether this backend's annotation library can be imported here."""
        return False

    @classmethod
    def detect(cls) -> Tracer:
        """The best available tracer: one matching a present GPU, else any, else no-op."""
        backends = [b for b in cls.implementations() if b.is_available()]
        present = {gpu.vendor for gpu in GPU.all()}
        for backend in backends:
            if backend.vendor in present:
                return backend()
        return backends[0]() if backends else cls()

    def push(self, name: str) -> None:
        """Open a named range on the native timeline (no-op in the base)."""

    def pop(self) -> None:
        """Close the most recently opened range."""

    def mark(self, name: str) -> None:
        """Emit an instantaneous named event."""

    def supported(self) -> Activity:
        """The :class:`Activity` kinds this backend can collect on the current device.

        Support is device- and driver-specific (e.g. consumer GPUs lack some CUPTI
        kinds), so a backend probes the hardware. The base supports none — it has no
        deep trace — which makes :meth:`collect` a silent no-op rather than an error
        on a host with no profiling backend.
        """
        return Activity(0)

    def collect(self, kinds: Activity = Activity.DEFAULT) -> TraceCollector:
        """A deep per-op trace collector for ``kinds``, resolved against device support.

        ``Activity.ALL`` means "everything this device offers", so it *adapts* down to
        the supported subset (dropped kinds are logged). Any *explicitly* requested kind
        the device cannot collect *fails fast* with :class:`ValueError` — better a clear
        error than a profile that silently omits what you asked for.
        """
        return self.open(self.resolve(kinds))

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

    def open(self, kinds: Activity) -> TraceCollector:
        """Build the collector for already-resolved ``kinds`` (no-op base; backends override)."""
        return TraceCollector()

    def timestamp(self) -> int:
        """Device-clock timestamp (ns) for region binning; host clock in the base."""
        return time.perf_counter_ns()

    def callbacks(self, domains: tuple[str, ...] = ("runtime", "driver")) -> CallbackSession:
        """A synchronous API-call callback session (no-op base; vendor backends override).

        domains: which callback domains to subscribe to (``runtime``/``driver``/``nvtx``).
        """
        return CallbackSession()
