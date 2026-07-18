import importlib
import logging
import os
import sys
import threading
from collections import deque
from contextlib import ExitStack
from contextvars import ContextVar
from dataclasses import dataclass, field
from enum import Flag, auto
from pathlib import Path
from tempfile import TemporaryDirectory
from types import CodeType, FunctionType, ModuleType, TracebackType
from typing import TYPE_CHECKING

from ..gpu import GPU
from . import annotate
from .models import RegionStat, RegionSummary
from .python import AsyncMode, PythonFormat, PythonMode, Tachyon
from .result import Profile
from .spans import activate, deactivate
from .target import Target
from .trace import Activity as NativeActivity
from .trace import BottleneckReport, RegionWindow, TraceCollector
from .tracer import Marker, Tracer

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..models.gpu_snapshot import GPUSnapshot


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class SpanFrame:
    """One live span and the evidence attributed to it."""

    name: str
    path: str
    thread: int
    device_start_ns: int
    finish_marker: Marker | None
    samples: deque[GPUSnapshot] = field(default_factory=lambda: deque(maxlen=4096))


@dataclass(frozen=True, slots=True)
class SpanMeasurement:
    """Raw span data kept cheap until a result is requested."""

    name: str
    wall_ms: float
    samples: tuple[GPUSnapshot, ...]


class Profiler:
    """Collect selected evidence through one bounded profiling session.

    `span` annotations stay dormant until this context is active. `features` controls
    what may be collected while the resulting `Profile` contains only evidence that
    was actually observed. Python sampling applies to `run`, `attach`, and `dump`.
    """

    class Feature(Flag):
        """Independent collection costs that may be combined with `|`."""

        PYTHON = auto()
        SPANS = auto()
        DEVICE = auto()
        MARKERS = auto()
        ACTIVITY = auto()
        DEFAULT = PYTHON | SPANS | DEVICE | MARKERS | ACTIVITY

    Activity = NativeActivity

    def __init__(
        self,
        *,
        features: Profiler.Feature = Feature.DEFAULT,
        activities: NativeActivity = NativeActivity.DEFAULT,
        device_index: int = 0,
        sample_interval_ms: int = 50,
        max_spans: int = 100_000,
        auto: Sequence[str] = (),
    ) -> None:
        self.features = features
        self.activities = activities
        self.device_index = device_index
        self.sample_interval_ms = sample_interval_ms
        self.max_spans = max_spans
        self.auto_modules = tuple(auto)
        self.gpu: GPU | None = None
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
        gpus = GPU.all() if self.features & (self.Feature.DEVICE | self.Feature.ACTIVITY) else ()
        if gpus:
            self.gpu = gpus[self.device_index] if self.device_index < len(gpus) else gpus[0]
        if self.features & (self.Feature.MARKERS | self.Feature.ACTIVITY):
            self.tracer = annotate.tracer()
        with ExitStack() as rollback:
            activate(self)
            rollback.callback(deactivate, self)
            if self.features & self.Feature.ACTIVITY and self.gpu is not None:
                self.collector = rollback.enter_context(self.tracer.collect(self.activities))
            if self.auto_modules:
                self.auto(self.auto_modules)
                rollback.callback(annotate.disable_auto)
            if self.features & self.Feature.DEVICE and self.gpu is not None:
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
        if self.features & self.Feature.ACTIVITY and self.gpu is not None:
            self.collector.stop()
        self.active = False

    def stop_sampler(self) -> None:
        """Stop and release this session's optional device sampler."""
        self.stop_event.set()
        if self.sampler is not None:
            self.sampler.join(timeout=2.0)
            self.sampler = None

    def enter(self, name: str) -> int:
        """Open one span and return the exact token later used to close it."""
        stack = self.stack.get()
        finish_marker = self.tracer.start(name) if self.features & self.Feature.MARKERS else None
        with self.lock:
            self.next_token += 1
            token = self.next_token
            parents = [self.frames[parent].name for parent in stack if parent in self.frames]
            self.frames[token] = SpanFrame(
                name=name,
                path=".".join((*parents, name)),
                thread=threading.get_ident(),
                device_start_ns=self.tracer.timestamp()
                if self.features & self.Feature.ACTIVITY
                else 0,
                finish_marker=finish_marker,
            )
        self.stack.set((*stack, token))
        return token

    def exit(self, token: int, wall_ns: int) -> None:
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
        if not samples and self.features & self.Feature.DEVICE:
            boundary = self.target_snapshot(frame.path)
            samples = [boundary] if boundary is not None else []
        if self.features & self.Feature.SPANS or samples:
            if len(self.measurements) == self.max_spans:
                self.dropped_spans += 1
            self.measurements.append(
                SpanMeasurement(
                    name=frame.path,
                    wall_ms=wall_ns / 1e6,
                    samples=tuple(samples),
                )
            )
        if self.features & self.Feature.ACTIVITY:
            if len(self.windows) == self.max_spans:
                self.dropped_spans += 1
            self.windows.append(
                RegionWindow(
                    name=frame.path,
                    start_ns=frame.device_start_ns,
                    end_ns=self.tracer.timestamp(),
                    wall_ns=wall_ns,
                )
            )

    def target_snapshot(self, name: str) -> GPUSnapshot | None:
        """Read one GPU snapshot only when it contains this process."""
        gpu = self.gpu
        if gpu is None:
            return None
        try:
            snapshot = gpu.snapshot(name=name)
        except (OSError, RuntimeError):
            logger.warning("device sampler skipped a failed snapshot", exc_info=True)
            return None
        process = next((item for item in snapshot.processes if item.pid == os.getpid()), None)
        if process is None:
            return None
        self.gpu_evidence = True
        self.gpu_label = snapshot.unit_name
        memory = snapshot.memory.model_copy(
            update={"scope": "process", "used_bytes": process.used_bytes}
        )
        return snapshot.model_copy(update={"memory": memory, "processes": [process]})

    def sample(self) -> None:
        """Poll target-process device telemetry while at least one span is open."""
        interval = self.sample_interval_ms / 1000.0
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

    def auto(self, modules: Sequence[str]) -> None:
        """Enable local `sys.monitoring` events only for code owned by `modules`."""
        annotate.enable_auto(self.module_codes(modules))
        self.auto_on = True

    @staticmethod
    def module_codes(modules: Sequence[str]) -> set[CodeType]:
        """Find owned module and nested code objects for local PEP 669 events."""
        loaded = [importlib.import_module(name) for name in modules]
        roots = tuple(
            str(Path(file).parent)
            for module in loaded
            if (file := getattr(module, "__file__", None))
        )
        found: set[CodeType] = set()
        pending = {code for module in loaded for code in Profiler.owned_codes(module, roots)}
        while pending:
            code = pending.pop()
            found.add(code)
            pending.update(
                item for item in code.co_consts if isinstance(item, CodeType) and item not in found
            )
        return found

    @staticmethod
    def owned_codes(module: ModuleType, roots: tuple[str, ...]) -> tuple[CodeType, ...]:
        """Return function code owned by selected roots, including class methods."""
        functions = (value for value in vars(module).values() if isinstance(value, FunctionType))
        classes = (value for value in vars(module).values() if isinstance(value, type))
        methods = (
            member
            for cls in classes
            for member in vars(cls).values()
            if isinstance(member, FunctionType)
        )
        return tuple(
            function.__code__
            for function in (*functions, *methods)
            if function.__code__.co_filename.startswith(roots)
        )

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

    def stats(self) -> list[RegionStat]:
        """Return per-span aggregates for the current session."""
        return self.result().stats()

    def bottlenecks(self, top: int = 10) -> list[RegionStat]:
        """Return the slowest span paths in the current session."""
        return self.result().bottlenecks(top)

    def trace_report(self, top: int = 10) -> BottleneckReport:
        """Return GPU activity attributed to span windows."""
        return self.result().trace_report(top)

    def report(self) -> str:
        """Render the current result as plain text."""
        return self.result().report()

    def show(self, *, color: bool = True) -> None:
        """Print the current result."""
        self.result().show(color=color)

    @classmethod
    def run(
        cls,
        target: str,
        *,
        module: bool | None = None,
        args: tuple[str, ...] = (),
        features: Profiler.Feature = Feature.DEFAULT,
        activities: NativeActivity = NativeActivity.DEFAULT,
        mode: PythonMode = PythonMode.WALL,
        format: PythonFormat = PythonFormat.PSTATS,
        output: str | None = None,
        duration: float | None = None,
        sampling_rate: str = "1khz",
        executable: str = sys.executable,
        timeout: float | None = None,
        strict: bool = False,
    ) -> Profile:
        """Run one target once and collect every selected capability that works."""
        target_spec = Target.resolve(target, module=module, args=args)
        tachyon = Tachyon(
            executable=Path(executable),
            mode=mode,
            format=format,
            output=Path(output) if output else None,
            duration=duration,
            sampling_rate=sampling_rate,
        )
        wants_python = bool(features & cls.Feature.PYTHON)
        python_available = wants_python and tachyon.available()
        if strict and wants_python and not python_available:
            raise RuntimeError("Python sampling requires Python 3.15")
        local_features = features & ~cls.Feature.PYTHON
        if local_features:
            return cls.run_instrumented(
                target_spec,
                tachyon=tachyon if python_available else None,
                features=local_features,
                activities=activities,
                timeout=timeout,
            )
        if python_available:
            return Profile(
                python=tachyon.run(
                    target_spec.name,
                    module=target_spec.module,
                    args=target_spec.args,
                    timeout=timeout,
                )
            )
        target_spec.run()
        return Profile()

    @classmethod
    def run_instrumented(
        cls,
        target: Target,
        *,
        tachyon: Tachyon | None,
        features: Profiler.Feature,
        activities: NativeActivity,
        timeout: float | None,
    ) -> Profile:
        """Run the target once with local collectors and an optional Tachyon parent."""
        with TemporaryDirectory(prefix="mainboard-") as directory:
            profile_path = Path(directory) / "profile.json"
            arguments = (
                "module" if target.module else "script",
                str(features.value),
                str(activities.value),
                str(profile_path),
                target.name,
                *target.args,
            )
            if tachyon is None:
                Target(name="mainboard.profiling.runner", module=True, args=arguments).run()
                return Profile.load(profile_path)
            python_profile = tachyon.run(
                "mainboard.profiling.runner",
                module=True,
                args=arguments,
                timeout=timeout,
            ).model_copy(update={"target": target.name})
            return Profile.load(profile_path).model_copy(update={"python": python_profile})

    @staticmethod
    def attach(
        pid: int,
        *,
        mode: PythonMode = PythonMode.WALL,
        format: PythonFormat = PythonFormat.PSTATS,
        output: str | None = None,
        duration: float | None = 30.0,
        sampling_rate: str = "1khz",
        executable: str = sys.executable,
        timeout: float | None = None,
    ) -> Profile:
        """Attach Python sampling to one live process."""
        python_profile = Tachyon(
            executable=Path(executable),
            mode=mode,
            format=format,
            output=Path(output) if output else None,
            duration=duration,
            sampling_rate=sampling_rate,
        ).attach(pid, timeout=timeout)
        return Profile(python=python_profile)

    @staticmethod
    def dump(
        pid: int,
        *,
        all_threads: bool = True,
        async_aware: bool = False,
        executable: str = sys.executable,
        timeout: float | None = 10.0,
    ) -> Profile:
        """Return one sampled Python stack snapshot from a live process."""
        python_profile = Tachyon(
            executable=Path(executable),
            all_threads=all_threads,
            async_aware=async_aware,
            async_mode=AsyncMode.ALL,
        ).dump(pid, timeout=timeout)
        return Profile(python=python_profile)
