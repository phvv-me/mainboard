import importlib
import logging
import os
import threading
import time
from collections import deque
from collections.abc import (
    Callable,
    Sequence,  # noqa: TC003  reason=Profiler is inspect.signature()'d in tests, so __init__'s Sequence[...] annotations must resolve at runtime since=2026-08-17
)
from contextlib import ExitStack
from contextvars import ContextVar
from types import CodeType, FunctionType, ModuleType, TracebackType
from typing import TypeAlias

# The one place profiling reaches the probe package, to discover the host's devices. The narrow
# `probe.units.gpu` imports nothing from here, so `probe.gating` -> `profile.bottleneck` cannot
# close into a cycle.
from ..probe.units.gpu import GPU
from . import annotate
from .models import ProcessReading, RegionStat, RegionSummary
from .protocols import (
    DeviceProbe,  # noqa: TC001  reason=Profiler is inspect.signature()'d in tests, so __init__'s Sequence[DeviceProbe] annotation must resolve at runtime since=2026-08-17
)
from .result import DeviceEvidence, Profile
from .session import Collection as Collection
from .session import Feature as Feature
from .session import SpanFrame as SpanFrame
from .session import SpanMeasurement as SpanMeasurement
from .spans import activate, active, deactivate
from .trace import Activity as NativeActivity
from .trace import BottleneckReport, RegionWindow, TraceCollector
from .tracer import Tracer

logger = logging.getLogger(__name__)


class Profiler:
    """Collect selected evidence through one bounded profiling session.

    `span` annotations stay dormant until this context is active. `features` controls what may
    be collected; the resulting `Profile` holds only evidence actually observed.
    """

    # Not a PEP 695 `type`, whose `TypeAliasType` would not forward `Profiler.Feature.SPANS`.
    Feature: TypeAlias = Feature  # noqa: UP040  reason=type statement would not forward attribute access since=2026-08-16
    Activity = NativeActivity

    def __init__(
        self,
        *,
        gpus: Sequence[DeviceProbe] = (),
        features: Feature = Feature.DEFAULT,
        activities: NativeActivity = NativeActivity.DEFAULT,
        device_index: int = 0,
        sample_interval_ms: int = 50,
        max_spans: int = 100_000,
        auto: Sequence[str] = (),
    ) -> None:
        self.collection = Collection(
            features=features,
            activities=activities,
            device_index=device_index,
            sample_interval_ms=sample_interval_ms,
            max_spans=max_spans,
            auto=tuple(auto),
        )
        self.gpus: Sequence[DeviceProbe] = tuple(gpus)
        self.gpu: DeviceProbe | None = None
        self.gpu_label = ""
        self.tracer: Tracer = Tracer()
        self.measurements: deque[SpanMeasurement] = deque(maxlen=max_spans)
        self.frames: dict[int, SpanFrame] = {}
        self.windows: deque[RegionWindow] = deque(maxlen=max_spans)
        self.collector: TraceCollector = TraceCollector()
        self.stack: ContextVar[tuple[int, ...]] = ContextVar("mainboard_spans", default=())
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.sampler: threading.Thread | None = None
        self.next_token = 0
        self.dropped_spans = 0
        self.gpu_evidence = False
        self.auto_on = False
        self.active = False

    def __enter__(self) -> Profiler:
        if self.active:
            raise RuntimeError("a Profiler instance cannot be entered twice")
        wanted = self.collection.features
        if wanted & (self.Feature.DEVICE | self.Feature.ACTIVITY):
            # Handed no probe, discover the host's own cards rather than collecting nothing.
            self.gpus = self.gpus or GPU.all()
            self.gpu = self._selected()
        if self.gpu is None and self._demands_activity():
            raise RuntimeError(
                "GPU activity collection was requested and no device is visible here, so "
                "this session would collect nothing. Run `mainboard facts` to see what the "
                "host probe finds, or drop `Profiler.Feature.ACTIVITY` to profile the host "
                "alone."
            )
        if wanted & (self.Feature.MARKERS | self.Feature.ACTIVITY):
            present = frozenset(gpu.vendor for gpu in self.gpus)
            self.tracer = annotate.tracer(present=present)
        with ExitStack() as rollback:
            activate(self)
            rollback.callback(deactivate, self)
            if wanted & self.Feature.ACTIVITY and self.gpu is not None:
                kinds = self.collection.activities
                self.collector = rollback.enter_context(self.tracer.collect(kinds))
            if self.collection.auto:
                self.auto(self.collection.auto)
                rollback.callback(annotate.disable_auto)
            if wanted & self.Feature.DEVICE and self.gpu is not None:
                self.stop_event.clear()
                self.sampler = threading.Thread(
                    target=self.sample, daemon=True, name="mainboard-profiler"
                )
                self.sampler.start()
                rollback.callback(self.stop_sampler)
            self.active = True
            rollback.pop_all()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None = None,
        exc: BaseException | None = None,
        traceback: TracebackType | None = None,
    ) -> None:
        if self.auto_on:
            annotate.disable_auto()
            self.auto_on = False
        deactivate(self)
        self.stop_sampler()
        try:
            if self.collection.features & self.Feature.ACTIVITY and self.gpu is not None:
                self.collector.stop()
        finally:
            self.active = False

    @classmethod
    def capture[Answer](
        cls,
        work: Callable[[], Answer],
        *,
        activities: NativeActivity = NativeActivity.DEFAULT,
        device_index: int = 0,
    ) -> tuple[Answer, Profile]:
        """Run once inside a synchronized, single-context native activity window.

        Reuses the active activity owner or opens one; nested windows are views of its records,
        never new subscribers or additions to its physical totals. All CUDA streams in the
        current context are synchronized, and CUDA must be issued serially from one host
        thread: unrelated concurrent CUDA work is not attributed.
        """
        if not activities or activities & ~NativeActivity.DEFAULT:
            raise ValueError("synchronized windows currently support kernels and memory copies")
        owner = active()
        if owner is not None:
            if not isinstance(owner, cls):
                raise RuntimeError("the active span owner cannot provide native activity windows")
            if owner.collection.device_index != device_index:
                raise RuntimeError("activity window requested a different CUDA-visible device")
            return owner._capture(work, activities)
        with cls(
            features=Feature.ACTIVITY, activities=activities, device_index=device_index
        ) as owner:
            return owner._capture(work, activities)

    def _capture[Answer](
        self, work: Callable[[], Answer], activities: NativeActivity
    ) -> tuple[Answer, Profile]:
        """Checkpoint the shared stream, run once, then retrieve exactly that interval."""
        if not self.active or not self.collection.features & Feature.ACTIVITY:
            raise RuntimeError("the active Profiler did not request native activity collection")
        if activities & ~self.collection.activities:
            raise RuntimeError("the active Profiler did not enable the requested activity kinds")
        if self.collector.device_index != self.collection.device_index:
            raise RuntimeError("selected device differs from the actual current CUDA device")
        since, loss_before = self.collector.checkpoint(activities)
        start_ns = self.tracer.timestamp()
        began = time.perf_counter_ns()
        try:
            answer = work()
        finally:
            until, loss_after = self.collector.checkpoint(activities)
        wall_ns = time.perf_counter_ns() - began
        end_ns = self.tracer.timestamp()
        if loss_after != loss_before:
            raise RuntimeError(f"native activity window lost {loss_after - loss_before} records")
        kernels = tuple(self.collector.kernels(since=since, until=until))
        memcpys = tuple(self.collector.memcpys(since=since, until=until))
        kernels = kernels if activities & NativeActivity.KERNEL else ()
        memcpys = memcpys if activities & NativeActivity.MEMCPY else ()
        observed = bool(kernels or memcpys)
        name = getattr(work, "__qualname__", type(work).__qualname__)
        return answer, Profile(
            device=self.gpu.label if self.gpu is not None else "",
            device_evidence=DeviceEvidence.COLLECTED if observed else DeviceEvidence.ABSENT,
            kernels=kernels,
            memcpys=memcpys,
            windows=(RegionWindow(name=name, start_ns=start_ns, end_ns=end_ns, wall_ns=wall_ns),),
        )

    @staticmethod
    def module_codes(modules: Sequence[str]) -> set[CodeType]:
        """Find owned module and nested code objects for local PEP 669 events."""
        loaded = [importlib.import_module(name) for name in modules]
        found: set[CodeType] = set()
        pending = {code for module in loaded for code in Profiler.owned_codes(module)}
        while pending:
            code = pending.pop()
            found.add(code)
            pending.update(
                item for item in code.co_consts if isinstance(item, CodeType) and item not in found
            )
        return found

    @staticmethod
    def owned_codes(module: ModuleType) -> tuple[CodeType, ...]:
        """Return function code owned by one module, including its class methods."""
        owned = [
            value
            for value in vars(module).values()
            if isinstance(value, FunctionType | type) and value.__module__ == module.__name__
        ]
        functions = [value for value in owned if isinstance(value, FunctionType)]
        methods = [
            member
            for cls in owned
            if isinstance(cls, type)
            for member in vars(cls).values()
            if isinstance(member, FunctionType)
        ]
        return tuple(function.__code__ for function in (*functions, *methods))

    @classmethod
    def under(cls, collection: Collection, *, gpus: Sequence[DeviceProbe] = ()) -> Profiler:
        """Build a profiler from one collection policy, for a caller that holds one already.

        The constructor takes the choices flat for a one-line caller; a study hands the value
        over rather than risk unpacking it differently next time.
        """
        return cls(
            gpus=gpus,
            features=collection.features,
            activities=collection.activities,
            device_index=collection.device_index,
            sample_interval_ms=collection.sample_interval_ms,
            max_spans=collection.max_spans,
            auto=collection.auto,
        )

    def auto(self, modules: Sequence[str]) -> None:
        """Enable local `sys.monitoring` events only for code owned by `modules`."""
        annotate.enable_auto(self.module_codes(modules))
        self.auto_on = True

    def bottlenecks(self, top: int = 10) -> list[RegionStat]:
        """Return the slowest span paths in the current session."""
        return self.result().bottlenecks(top)

    def _demands_activity(self) -> bool:
        """Whether this session named ACTIVITY, which a GPU-less host cannot serve.

        Under DEFAULT such a host should still profile its Python.
        """
        wanted = self.collection.features
        return bool(wanted & self.Feature.ACTIVITY) and wanted != self.Feature.DEFAULT

    def enter(self, name: str) -> int:
        """Open one span and return the exact token later used to close it."""
        stack = self.stack.get()
        marking = self.collection.features & self.Feature.MARKERS
        finish_marker = self.tracer.start(name) if marking else None
        with self.lock:
            self.next_token += 1
            token = self.next_token
            parents = [self.frames[parent].name for parent in stack if parent in self.frames]
            self.frames[token] = SpanFrame(
                name=name,
                path=".".join((*parents, name)),
                thread=threading.get_ident(),
                device_start_ns=self.tracer.timestamp()
                if self.collection.features & self.Feature.ACTIVITY
                else 0,
                finish_marker=finish_marker,
            )
        self.stack.set((*stack, token))
        return token

    def _evidence(self, *, observed: bool) -> DeviceEvidence:
        """Say whether device evidence came back, or was never asked for in the first place."""
        if observed:
            return DeviceEvidence.COLLECTED
        wanted = self.collection.features & (self.Feature.DEVICE | self.Feature.ACTIVITY)
        return DeviceEvidence.ABSENT if wanted else DeviceEvidence.UNSOUGHT

    def exit(self, token: int, *, wall_ns: int) -> None:
        """Close one span and fold its timing, device samples, and activity window."""
        stack = self.stack.get()
        if token in stack:
            self.stack.set(tuple(item for item in stack if item != token))
        with self.lock:
            frame = self.frames.pop(token, None)
        if frame is None:
            return
        if frame.finish_marker is not None:
            frame.finish_marker()
        samples = list(frame.samples)
        if not samples and self.collection.features & self.Feature.DEVICE:
            boundary = self.target_snapshot(frame.path)
            samples = [boundary] if boundary is not None else []
        if self.collection.features & self.Feature.SPANS or samples:
            if len(self.measurements) == self.collection.max_spans:
                self.dropped_spans += 1
            self.measurements.append(
                SpanMeasurement(
                    name=frame.path,
                    wall_ms=wall_ns / 1e6,
                    samples=tuple(samples),
                )
            )
        if self.collection.features & self.Feature.ACTIVITY:
            if len(self.windows) == self.collection.max_spans:
                self.dropped_spans += 1
            self.windows.append(
                RegionWindow(
                    name=frame.path,
                    start_ns=frame.device_start_ns,
                    end_ns=self.tracer.timestamp(),
                    wall_ns=wall_ns,
                )
            )

    def report(self) -> str:
        """Render the current result as plain text."""
        return self.result().report()

    def result(self) -> Profile:
        """Freeze the evidence collected so far into one `Profile`."""
        kernels = tuple(self.collector.kernels())
        memcpys = tuple(self.collector.memcpys())
        activities = tuple(self.collector.activities())
        used_gpu = self.gpu_evidence or bool(kernels or memcpys or activities)
        return Profile(
            device=self.gpu_label
            if self.gpu_evidence
            else (self.gpu.label if used_gpu and self.gpu is not None else ""),
            device_evidence=self._evidence(observed=used_gpu),
            summaries=tuple(
                RegionSummary.from_snaps(item.name, item.wall_ms, item.samples)
                for item in self.measurements
            ),
            windows=tuple(self.windows),
            kernels=kernels,
            memcpys=memcpys,
            activities=activities,
            dropped_spans=self.dropped_spans,
            dropped_activities=self.collector.dropped(),
        )

    def sample(self) -> None:
        """Poll target-process device telemetry while at least one span is open."""
        interval = self.collection.sample_interval_ms / 1000.0
        while not self.stop_event.wait(interval):
            with self.lock:
                frames = tuple(self.frames.values())
                name = frames[-1].path if frames else ""
            if not frames:
                continue
            snapshot = self.target_snapshot(name)
            if snapshot is None:
                continue
            with self.lock:
                for frame in frames:
                    frame.samples.append(snapshot)

    def _selected(self) -> DeviceProbe | None:
        """Select exactly the requested visible device; never substitute another card."""
        if not self.gpus:
            return None
        index = self.collection.device_index
        try:
            return self.gpus[index]
        except IndexError:
            raise ValueError(
                f"device_index {index} is outside the {len(self.gpus)} visible devices"
            ) from None

    def stats(self) -> list[RegionStat]:
        """Return per-span aggregates for the current session."""
        return self.result().stats()

    def stop_sampler(self) -> None:
        """Stop and release this session's optional device sampler."""
        self.stop_event.set()
        if self.sampler is not None:
            self.sampler.join(timeout=2.0)
            self.sampler = None

    def target_snapshot(self, name: str) -> ProcessReading | None:
        """Read one process-scoped device reading, only when it contains this process."""
        gpu = self.gpu
        if gpu is None:
            return None
        try:
            raw = gpu.snapshot(name=name)
        except OSError, RuntimeError:
            logger.warning("device sampler skipped a failed snapshot", exc_info=True)
            return None
        process = next((item for item in raw.processes if item.pid == os.getpid()), None)
        if process is None:
            return None
        self.gpu_evidence = True
        self.gpu_label = raw.unit_name
        return ProcessReading(
            unit_name=raw.unit_name,
            memory_used_bytes=process.used_bytes,
            gpu_util_pct=raw.utilization.gpu_pct,
            memory_util_pct=raw.utilization.memory_pct,
            power_w=raw.energy.power_w,
            temperature_c=raw.thermal.temperature_c,
        )

    def trace_report(self, top: int = 10) -> BottleneckReport:
        """Return GPU activity attributed to span windows."""
        return self.result().trace_report(top)
