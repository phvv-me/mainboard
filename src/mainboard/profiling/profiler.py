import importlib
import logging
import os
import sys
import threading
from collections import deque
from collections.abc import Sequence
from contextlib import ExitStack
from contextvars import ContextVar
from dataclasses import dataclass, field
from enum import Flag, auto
from pathlib import Path
from tempfile import TemporaryDirectory
from types import CodeType, FunctionType, ModuleType, TracebackType
from typing import TYPE_CHECKING, TypeAlias

from ..gpu import GPU
from ..models.base import FrozenModel
from . import annotate
from .models import RegionStat, RegionSummary
from .python import AsyncMode, Tachyon
from .result import Profile
from .spans import activate, deactivate
from .target import Target
from .trace import Activity as NativeActivity
from .trace import BottleneckReport, RegionWindow, TraceCollector
from .tracer import Marker, Tracer

if TYPE_CHECKING:
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
    samples: deque["GPUSnapshot"] = field(default_factory=lambda: deque(maxlen=4096))


@dataclass(frozen=True, slots=True)
class SpanMeasurement:
    """Raw span data kept cheap until a result is requested."""

    name: str
    wall_ms: float
    samples: tuple["GPUSnapshot", ...]


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
    device_index: which GPU to sample.
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

    @classmethod
    def here(cls) -> "Reach":
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
    ) -> "Reach":
        """Run one script or module once and measure it."""
        return cls(target=target, module=module, args=args, timeout=timeout)

    @classmethod
    def attaching(cls, pid: int, *, timeout: float | None = None) -> "Reach":
        """Attach to a process already running."""
        return cls(pid=pid, timeout=timeout)

    @property
    def kind(self) -> str:
        """Return which of the three this is, for a row key or an error message."""
        if self.pid:
            return "attach"
        return "launch" if self.target else "here"


class Profiler:
    """Collect selected evidence through one bounded profiling session.

    `span` annotations stay dormant until this context is active. `features` controls
    what may be collected while the resulting `Profile` contains only evidence that
    was actually observed. Python sampling applies to `run`, `attach`, and `dump`.
    """

    # `TypeAlias` tells mypy this is a type, not a plain variable, so a method below that
    # annotates a parameter as bare `Feature` still resolves to the module-level enum instead
    # of this class attribute of the same name. The PEP 695 `type` statement ruff suggests here
    # would wrap `Feature` in a `TypeAliasType`, which does not forward attribute access, so
    # `Profiler.Feature.SPANS` would break at runtime for every caller.
    Feature: TypeAlias = Feature  # noqa: UP040
    Activity = NativeActivity

    def __init__(
        self,
        *,
        features: Feature = Feature.DEFAULT,
        activities: NativeActivity = NativeActivity.DEFAULT,
        device_index: int = 0,
        sample_interval_ms: int = 50,
        max_spans: int = 100_000,
        auto: Sequence[str] = (),
    ) -> None:
        # The constructor stays a flat façade over the model, the way the command line is a flat
        # façade over `Tachyon`. One object is what a study passes around; loose keywords are
        # what a caller writing one line wants.
        self.collection = Collection(
            features=features,
            activities=activities,
            device_index=device_index,
            sample_interval_ms=sample_interval_ms,
            max_spans=max_spans,
            auto=tuple(auto),
        )
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

    @classmethod
    def under(cls, collection: Collection) -> "Profiler":
        """Build a profiler from one collection policy.

        The constructor takes the six choices flat because that is what a caller writing one line
        wants. Anything holding a policy already, a study most of all, should hand over the value
        rather than unpack it into six arguments and risk unpacking it differently next time.
        """
        return cls(
            features=collection.features,
            activities=collection.activities,
            device_index=collection.device_index,
            sample_interval_ms=collection.sample_interval_ms,
            max_spans=collection.max_spans,
            auto=collection.auto,
        )

    def __enter__(self) -> "Profiler":
        if self.active:
            raise RuntimeError("a Profiler instance cannot be entered twice")
        wanted = self.collection.features
        index = self.collection.device_index
        gpus = GPU.all() if wanted & (self.Feature.DEVICE | self.Feature.ACTIVITY) else ()
        if gpus:
            self.gpu = gpus[index] if index < len(gpus) else gpus[0]
        if wanted & (self.Feature.MARKERS | self.Feature.ACTIVITY):
            self.tracer = annotate.tracer()
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

    def stop_sampler(self) -> None:
        """Stop and release this session's optional device sampler."""
        self.stop_event.set()
        if self.sampler is not None:
            self.sampler.join(timeout=2.0)
            self.sampler = None

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

    def target_snapshot(self, name: str) -> "GPUSnapshot | None":
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

    def auto(self, modules: Sequence[str]) -> None:
        """Enable local `sys.monitoring` events only for code owned by `modules`."""
        annotate.enable_auto(self.module_codes(modules))
        self.auto_on = True

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
    def measure(
        cls,
        reach: Reach,
        *,
        collection: Collection | None = None,
        sampler: Tachyon | None = None,
        strict: bool = False,
    ) -> Profile:
        """Measure whatever `reach` names, under `collection`, driving `sampler`.

        One method for all three ways of reaching a target, because which one applies is a
        property of the `Reach` rather than a choice of function. `run` and `attach` remain as
        the two shorthands a caller writes by hand. `Reach.here()` names the calling process,
        which has no target for this classmethod to launch, so it raises rather than launching
        an empty target: that measurement is what `with Profiler(...) as profiler:` is for.
        """
        policy = collection or Collection()
        if reach.kind == "attach":
            tachyon = sampler or Tachyon(duration=30.0)
            tachyon.require_available()
            return Profile(python=tachyon.attach(reach.pid, timeout=reach.timeout))
        if reach.kind == "here":
            raise ValueError(
                "Reach.here() names the calling process, which measure() cannot launch; wrap "
                "that code in `with Profiler(...) as profiler:` instead."
            )
        return cls.run(
            reach.target,
            module=reach.module,
            args=reach.args,
            features=policy.features,
            activities=policy.activities,
            sampler=sampler,
            timeout=reach.timeout,
            strict=strict,
        )

    @classmethod
    def run(
        cls,
        target: str,
        *,
        module: bool | None = None,
        args: tuple[str, ...] = (),
        features: Feature = Feature.DEFAULT,
        activities: NativeActivity = NativeActivity.DEFAULT,
        sampler: Tachyon | None = None,
        timeout: float | None = None,
        strict: bool = False,
    ) -> Profile:
        """Run one target once and collect every selected capability that works.

        sampler: how to drive the external Python sampler, as the model that already describes
            it. Its `executable` is also the interpreter the target runs under, so the two
            cannot drift apart the way two separate arguments could.
        """
        target_spec = Target.resolve(target, module=module, args=args)
        tachyon = sampler or Tachyon()
        executable = str(tachyon.executable)
        wants_python = bool(features & cls.Feature.PYTHON)
        python_available = wants_python and tachyon.available()
        if strict and wants_python and not python_available:
            raise RuntimeError("Python sampling requires Python 3.15")
        local_features = features & ~cls.Feature.PYTHON
        if local_features:
            return cls.run_instrumented(
                target_spec,
                tachyon=tachyon if python_available else None,
                executable=Path(executable),
                features=local_features,
                activities=activities,
                timeout=timeout,
            )
        if python_available:
            return Profile(
                python=tachyon.run(
                    target_spec.name,
                    module=target_spec.module,
                    args=tuple(target_spec.args),
                    timeout=timeout,
                )
            )
        if Path(executable) == Path(sys.executable) and timeout is None:
            target_spec.run()
        else:
            target_spec.launch(Path(executable), timeout=timeout)
        return Profile()

    @classmethod
    def run_instrumented(
        cls,
        target: Target,
        *,
        tachyon: Tachyon | None,
        executable: Path,
        features: Feature,
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
                runner = Target(name="mainboard.profiling.runner", module=True, args=arguments)
                if executable == Path(sys.executable) and timeout is None:
                    runner.run()
                else:
                    runner.launch(
                        executable,
                        timeout=timeout,
                        import_paths=(Path(__file__).parents[2],),
                    )
                return Profile.load(profile_path)
            python_profile = tachyon.run(
                "mainboard.profiling.runner",
                module=True,
                args=arguments,
                timeout=timeout,
                import_paths=(Path(__file__).parents[2],),
            ).model_copy(update={"target": target.name})
            return Profile.load(profile_path).model_copy(update={"python": python_profile})

    @staticmethod
    def attach(
        pid: int,
        *,
        sampler: Tachyon | None = None,
        timeout: float | None = None,
    ) -> Profile:
        """Attach Python sampling to one live process.

        sampler: how to drive the external sampler. `Tachyon` already models every one of those
            choices, so this takes the model rather than taking its fields loose and rebuilding
            it, which is what let two call sites disagree about the same policy.
        """
        tachyon = sampler or Tachyon(duration=30.0)
        tachyon.require_available()
        python_profile = tachyon.attach(pid, timeout=timeout)
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
        tachyon = Tachyon(
            executable=Path(executable),
            all_threads=all_threads,
            async_aware=async_aware,
            async_mode=AsyncMode.ALL,
        )
        tachyon.require_available()
        python_profile = tachyon.dump(pid, timeout=timeout)
        return Profile(python=python_profile)
