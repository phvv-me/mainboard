import importlib
import logging
import os
import threading
from collections import deque
from collections.abc import (
    Sequence,  # noqa: TC003  reason=Profiler is inspect.signature()'d in tests, so __init__'s Sequence[...] annotations must resolve at runtime since=2026-08-17
)
from contextlib import ExitStack
from contextvars import ContextVar
from dataclasses import dataclass, field
from enum import Flag, auto
from types import CodeType, FunctionType, ModuleType, TracebackType
from typing import TypeAlias

from patos import FrozenModel

from . import annotate
from .models import ProcessReading, RegionStat, RegionSummary
from .protocols import (
    DeviceProbe,  # noqa: TC001  reason=Profiler is inspect.signature()'d in tests, so __init__'s Sequence[DeviceProbe] annotation must resolve at runtime since=2026-08-17
)
from .result import Profile
from .spans import activate, deactivate
from .trace import Activity as NativeActivity
from .trace import BottleneckReport, RegionWindow, TraceCollector
from .tracer import Marker, Tracer

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class SpanFrame:
    """One live span and the evidence attributed to it."""

    name: str
    path: str
    thread: int
    device_start_ns: int
    finish_marker: Marker | None
    samples: deque[ProcessReading] = field(default_factory=lambda: deque(maxlen=4096))


@dataclass(frozen=True, slots=True)
class SpanMeasurement:
    """Raw span data kept cheap until a result is requested."""

    name: str
    wall_ms: float
    samples: tuple[ProcessReading, ...]


class Feature(Flag):
    """Independent collection costs that may be combined with `|`.

    Independence is the contract, not a convenience: `DEFAULT` ORs every member because each
    one can be collected alongside the others without changing what the others observe. A
    capability that cannot honour that, such as hardware counter collection needing kernel
    replay, does not belong in this flag however convenient the syntax would be. It belongs in
    its own pass.
    """

    PYTHON = auto()
    SPANS = auto()
    DEVICE = auto()
    MARKERS = auto()
    ACTIVITY = auto()
    DEFAULT = PYTHON | SPANS | DEVICE | MARKERS | ACTIVITY


class Collection(FrozenModel):
    """What evidence to gather and at what cost, as one value.

    These six choices were six constructor arguments, two of which were then repeated on every
    other entry point, so the same policy had to be restated wherever it was wanted and nothing
    checked that two statements of it agreed. As one model it can be built once, passed around,
    compared and stored beside the measurement it produced, which is what a study needs: a
    throughput number whose collection policy is not attached to it is hard to reproduce.

    features: which capabilities may be collected.
    activities: which CUPTI activity kinds to keep when `ACTIVITY` is on.
    device_index: which of the constructor's `gpus` to sample.
    sample_interval_ms: how often to take a device telemetry sample.
    max_spans: the bound on retained spans and windows, past which they are dropped and counted.
    auto: modules whose functions are annotated automatically.
    """

    features: Feature = Feature.DEFAULT
    activities: NativeActivity = NativeActivity.DEFAULT
    device_index: int = 0
    sample_interval_ms: int = 50
    max_spans: int = 100_000
    auto: tuple[str, ...] = ()


class Reach(FrozenModel):
    """How to get at the thing being measured, as one value.

    There are exactly three ways and they were three entry points, which is why `executable` and
    `timeout` were spelled out on each of them. As one model the choice is data, so a study can
    hold it, vary it, or record which one produced a number, and one method serves all three.

    target: a script path or a module name. Empty means the calling process itself.
    module: whether `target` names a module rather than a path, or None to infer.
    args: arguments passed to a launched target.
    pid: a live process to attach to. Non-zero selects attachment and ignores `target`.
    timeout: how long to wait on a launched or attached target.
    """

    target: str = ""
    module: bool | None = None
    args: tuple[str, ...] = ()
    pid: int = 0
    timeout: float | None = None

    @property
    def kind(self) -> str:
        """Return which of the three this is, for a row key or an error message."""
        if self.pid:
            return "attach"
        return "launch" if self.target else "here"

    @classmethod
    def attaching(cls, pid: int, *, timeout: float | None = None) -> Reach:
        """Attach to a process already running."""
        return cls(pid=pid, timeout=timeout)

    @classmethod
    def here(cls) -> Reach:
        """Measure the calling process, which is what a `with` block does."""
        return cls()

    @classmethod
    def launch(
        cls,
        target: str,
        *,
        module: bool | None = None,
        args: tuple[str, ...] = (),
        timeout: float | None = None,
    ) -> Reach:
        """Run one script or module once and measure it.

        Launching a target and attaching to a live process both land with the Python
        sampling CLI layer; only `here()` (an active `Profiler` context) is wired yet.
        """
        return cls(target=target, module=module, args=args, timeout=timeout)


class Profiler:
    """Collect selected evidence through one bounded profiling session.

    `span` annotations stay dormant until this context is active. `features` controls
    what may be collected while the resulting `Profile` contains only evidence that
    was actually observed.
    """

    # `TypeAlias`, not PEP 695 `type` (ruff's suggestion), since a `type` statement wraps
    # `Feature` in a `TypeAliasType` that does not forward attribute access, breaking
    # `Profiler.Feature.SPANS` at runtime.
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
        # A flat façade over the model, the way the CLI is a flat façade over the collection
        # policy: loose keywords for a one-liner caller, one object for a study to pass around.
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
        index = self.collection.device_index
        gpus = self.gpus if wanted & (self.Feature.DEVICE | self.Feature.ACTIVITY) else ()
        if gpus:
            self.gpu = gpus[index] if index < len(gpus) else gpus[0]
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
        if self.collection.features & self.Feature.ACTIVITY and self.gpu is not None:
            self.collector.stop()
        self.active = False

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
        functions = (
            value
            for value in vars(module).values()
            if isinstance(value, FunctionType) and value.__module__ == module.__name__
        )
        classes = (
            value
            for value in vars(module).values()
            if isinstance(value, type) and value.__module__ == module.__name__
        )
        methods = (
            member
            for cls in classes
            for member in vars(cls).values()
            if isinstance(member, FunctionType)
        )
        return tuple(function.__code__ for function in (*functions, *methods))

    @classmethod
    def under(cls, collection: Collection, *, gpus: Sequence[DeviceProbe] = ()) -> Profiler:
        """Build a profiler from one collection policy.

        The constructor takes the six collection choices flat because that is what a caller
        writing one line wants. Anything holding a policy already, a study most of all, should
        hand over the value rather than unpack it into six arguments and risk unpacking it
        differently next time.
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

    @staticmethod
    def _skipped_snapshot() -> None:
        """Log one failed device snapshot as a warning and stand for its absent reading."""
        logger.warning("device sampler skipped a failed snapshot", exc_info=True)

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
            return self._skipped_snapshot()
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
