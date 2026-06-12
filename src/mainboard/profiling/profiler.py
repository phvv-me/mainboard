"""The profiler orchestrator: sample device telemetry while regions run.

Vendor-agnostic by construction — it samples the cross-platform
:meth:`GPU.snapshot` the rest of mainboard already provides, and the active
:class:`Tracer` (set up by :mod:`.annotate`) handles native annotation. A background
thread polls the device while `region`/`@profile`/auto-annotated calls bracket the
work; each region's snapshots aggregate into a :class:`RegionSummary`.
"""

from __future__ import annotations

import importlib
import logging
import threading
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING

from ..gpu import GPU
from . import annotate
from .models import RegionStat, RegionSummary
from .result import Profile
from .trace import Activity, BottleneckReport, RegionWindow, TraceCollector
from .tracer import Tracer

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..models.gpu_snapshot import GPUSnapshot


logger = logging.getLogger(__name__)


class _Frame:
    """A live region: its name, owning thread, device-clock start, and sampled snapshots."""

    def __init__(self, name: str, start_ns: int) -> None:
        self.name = name
        self.start_ns = start_ns
        self.thread = threading.get_ident()
        self.snaps: list[GPUSnapshot] = []


class Profiler:
    """Always-on sampler that times regions and aggregates device telemetry.

    device_index: which GPU from `GPU.all()` to sample. sample_interval_ms: polling
    period. trace: the deep per-kernel tier — ``False`` off, ``True`` for kernels +
    memcpy, or an :class:`Activity` flag for exactly those kinds (e.g.
    ``trace=Activity.ALL``). auto: packages to auto-annotate. Use as a context manager;
    annotate work with `region` / `@profile`.
    """

    def __init__(
        self,
        *,
        device_index: int = 0,
        sample_interval_ms: int = 50,
        trace: bool | Activity = False,
        auto: Sequence[str] = (),
    ) -> None:
        self.device_index = device_index
        self.sample_interval_ms = sample_interval_ms
        self.trace = trace
        self.trace_kinds = trace if isinstance(trace, Activity) else Activity.DEFAULT
        self.auto_modules = tuple(auto)
        self._gpu: GPU = GPU()
        self._tracer: Tracer = Tracer()
        self._summaries: list[RegionSummary] = []
        self._frames: list[_Frame] = []
        self._windows: list[RegionWindow] = []
        self._collector: TraceCollector = TraceCollector()
        self._auto_on = False
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> Profiler:
        gpus = GPU.all()
        if gpus:
            self._gpu = gpus[self.device_index] if self.device_index < len(gpus) else gpus[0]
        self._tracer = annotate.tracer()
        # collect() resolves kinds against device support: it may fail fast on an
        # explicitly-requested unsupported kind — do it before touching global state.
        if self.trace:
            self._collector = self._tracer.collect(self.trace_kinds).__enter__()
        annotate.profiler = self
        if self.auto_modules:
            self.auto(self.auto_modules)
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._sample, daemon=True, name="mainboard-profiler"
        )
        self._thread.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None = None,
        exc: BaseException | None = None,
        traceback: TracebackType | None = None,
    ) -> None:
        if self._auto_on:
            annotate.disable_auto()
            self._auto_on = False
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        annotate.profiler = None
        if self.trace:
            self._collector.stop()

    # ── region hooks (called by annotate._enter / _exit) ────────────────────────

    def enter(self, name: str) -> None:
        """Open a region frame so the sampler attributes snapshots to it."""
        start_ns = self._tracer.timestamp() if self.trace else 0
        with self._lock:
            self._frames.append(_Frame(name, start_ns))

    def exit(self, name: str, wall_ns: int) -> None:
        """Close the calling thread's innermost frame and fold its snapshots into a summary.

        Auto-annotation can fire in worker threads, so a frame is owned by the
        thread that opened it; popping the shared LIFO blindly would misattribute
        regions across threads.
        """
        with self._lock:
            frame = self._pop_thread_frame(threading.get_ident())
        if frame is None:
            return
        self._summaries.append(RegionSummary.from_snaps(name, wall_ns / 1e6, frame.snaps))
        if self.trace:
            self._windows.append(
                RegionWindow(
                    name=name,
                    start_ns=frame.start_ns,
                    end_ns=self._tracer.timestamp(),
                    wall_ns=wall_ns,
                )
            )

    def _pop_thread_frame(self, thread: int) -> _Frame | None:
        """Remove and return the innermost open frame owned by `thread` (caller holds the lock)."""
        for position in range(len(self._frames) - 1, -1, -1):
            if self._frames[position].thread == thread:
                return self._frames.pop(position)
        return None

    def _sample(self) -> None:
        interval = self.sample_interval_ms / 1000.0
        while not self._stop.wait(interval):
            with self._lock:
                frames = list(self._frames)
                name = frames[-1].name if frames else ""
            if not frames:
                continue
            # thread entry point: a transient sensor failure must not end sampling
            try:
                snap = self._gpu.snapshot(name=name)
            except Exception:
                logger.warning("sampler skipped a tick on a snapshot failure", exc_info=True)
                continue
            with self._lock:
                for frame in frames:
                    frame.snaps.append(snap)

    # ── auto-annotation ─────────────────────────────────────────────────────────

    def auto(self, modules: Sequence[str]) -> None:
        """Auto-annotate every call in ``modules`` at runtime (PEP 669, zero edits).

        For static source rewriting instead, use
        :func:`mainboard.profiling.instrument_source`. Disabled automatically when the
        profiler context exits.
        """
        roots = tuple(self._module_roots(modules))
        annotate.enable_auto(lambda code: code.co_filename.startswith(roots))
        self._auto_on = True

    @staticmethod
    def _module_roots(modules: Sequence[str]) -> list[str]:
        roots = []
        for name in modules:
            module = importlib.import_module(name)
            file = getattr(module, "__file__", None)
            if file:
                roots.append(str(Path(file).parent))
        return roots

    def supported(self) -> Activity:
        """The deep-trace :class:`Activity` kinds available on this machine's device.

        Probes the active backend so you can see what tracing this GPU offers before
        asking for it — ``Profiler().supported()`` returns e.g.
        ``Activity.KERNEL|MEMCPY|...``. Requesting a kind not in this set fails fast.
        """
        return annotate.tracer().supported()

    # ── results ─────────────────────────────────────────────────────────────────

    def result(self) -> Profile:
        """Snapshot the session into an immutable :class:`Profile` — the result value.

        Everything you do with a measurement (read, diff, export, show) is a method on
        :class:`Profile`; this is the only call that crosses from collection to result.
        """
        return Profile(
            device=self._gpu.name,
            summaries=tuple(self._summaries),
            windows=tuple(self._windows),
            kernels=tuple(self._collector.kernels()),
            memcpys=tuple(self._collector.memcpys()),
            activities=tuple(self._collector.activities()),
        )

    def summaries(self) -> list[RegionSummary]:
        """Every region occurrence, in completion order (the raw, un-collapsed log)."""
        return list(self._summaries)

    def stats(self) -> list[RegionStat]:
        """Per-name aggregates of the current result (see :meth:`Profile.stats`)."""
        return self.result().stats()

    def bottlenecks(self, top: int = 10) -> list[RegionStat]:
        """The slowest region names so far (see :meth:`Profile.bottlenecks`)."""
        return self.result().bottlenecks(top)

    def trace_report(self, top: int = 10) -> BottleneckReport:
        """Deep GPU-time ranking so far (see :meth:`Profile.trace_report`)."""
        return self.result().trace_report(top)

    def report(self) -> str:
        """Plain-text region table of the current result (see :meth:`Profile.report`)."""
        return self.result().report()

    def show(self, *, color: bool = True) -> None:
        """Print the rich region/kernel tables (see :meth:`Profile.show`)."""
        self.result().show(color=color)
