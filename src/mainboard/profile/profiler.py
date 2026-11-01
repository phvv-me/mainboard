import time
from collections.abc import (
    Callable,
    Sequence,  # noqa: TC003  reason=Profiler is inspect.signature()'d in tests, so __init__'s Sequence[...] annotations must resolve at runtime since=2026-08-17
)
from contextlib import ExitStack
from types import TracebackType
from typing import TypeAlias

# The one place profiling reaches the probe package, to discover the host's devices. The narrow
# `probe.units.gpu` imports nothing from here, so `probe.gating` -> `profile.bottleneck` cannot
# close into a cycle.
from ..probe.units.gpu import GPU
from .bottleneck import annotate
from .bottleneck.owned import module_codes
from .enums import Activity as NativeActivity
from .recording import DeviceSampler, SpanRecorder
from .result import DeviceEvidence, Profile
from .result.models import RegionSummary
from .session import Collection, Feature
from .spans import activate, active, deactivate
from .trace import KernelTrace, MemcpyTrace, RegionWindow, TraceCollector
from .tracer import Tracer

_MAX_SPANS = 100_000


class Profiler:
    """Collect selected evidence through one bounded profiling session.

    `span` annotations stay dormant until this context is active, then open and close in
    `spans`, while `sampler` reads the selected device. `features` controls what may be
    collected; the resulting `Profile` holds only evidence actually observed.
    """

    # Not a PEP 695 `type`, whose `TypeAliasType` would not forward `Profiler.Feature.SPANS`.
    Feature: TypeAlias = Feature  # noqa: UP040  reason=type statement would not forward attribute access since=2026-08-16
    Activity = NativeActivity

    active: bool
    auto_on: bool
    collection: Collection
    collector: TraceCollector
    gpus: Sequence[GPU]
    sampler: DeviceSampler
    spans: SpanRecorder

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
            return owner.capture_window(work, activities)
        with cls(
            features=Feature.ACTIVITY, activities=activities, device_index=device_index
        ) as owner:
            return owner.capture_window(work, activities)

    @classmethod
    def under(cls, collection: Collection, *, gpus: Sequence[GPU] = ()) -> Profiler:
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

    def __enter__(self) -> Profiler:
        if self.active:
            raise RuntimeError("a Profiler instance cannot be entered twice")
        wanted = self.collection.features
        if wanted & (self.Feature.DEVICE | self.Feature.ACTIVITY):
            # Handed no probe, discover the host's own cards rather than collecting nothing.
            self.gpus = self.gpus or GPU.all()
            self.sampler.gpu = self._selected()
        if self.sampler.gpu is None and self._demands_activity():
            raise RuntimeError(
                "GPU activity collection was requested and no device is visible here, so "
                "this session would collect nothing. Run `mb host list --facts` to see what the "
                "host probe finds, or drop `Profiler.Feature.ACTIVITY` to profile the host "
                "alone."
            )
        if wanted & (self.Feature.MARKERS | self.Feature.ACTIVITY):
            present = frozenset(gpu.vendor for gpu in self.gpus)
            self.spans.tracer = annotate.tracer(present=present)
        with ExitStack() as rollback:
            activate(self)
            rollback.callback(deactivate, self)
            self._armed(rollback, wanted)
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
        self.sampler.stop()
        try:
            self._stop_collector()
        finally:
            self.active = False

    def __init__(
        self,
        *,
        gpus: Sequence[GPU] = (),
        features: Feature = Feature.DEFAULT,
        activities: NativeActivity = NativeActivity.DEFAULT,
        device_index: int = 0,
        sample_interval_ms: int = 50,
        max_spans: int = _MAX_SPANS,
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
        self.gpus = tuple(gpus)
        self.auto_on = False
        self.active = False
        self.collector = TraceCollector()
        self.sampler = DeviceSampler(interval_ms=self.collection.sample_interval_ms)
        self.spans = SpanRecorder(self.collection, tracer=Tracer(), snapshot=self.sampler.snapshot)

    def _armed(self, rollback: ExitStack, wanted: Feature) -> None:
        """Start the collector, auto-instrumentation and sampler `wanted` asks.

        Each is undone by `rollback` should a later one fail.
        """
        gpu = self.sampler.gpu
        if wanted & self.Feature.ACTIVITY and gpu is not None:
            kinds = self.collection.activities
            self.collector = rollback.enter_context(self.spans.tracer.collect(kinds))
        if self.collection.auto:
            self._auto(self.collection.auto)
            rollback.callback(annotate.disable_auto)
        if wanted & self.Feature.DEVICE and gpu is not None:
            self.sampler.start(self.spans)
            rollback.callback(self.sampler.stop)

    def _auto(self, modules: Sequence[str]) -> None:
        """Enable local events for owned function bodies, leaving explicit spans active.

        Explicit annotations and automatic spans can nest, even with the same label. Named
        generators and coroutines emit active resume segments, not invocation counts.
        """
        annotate.enable_auto(module_codes(modules))
        self.auto_on = True

    def _capturable(self, activities: NativeActivity) -> None:
        """Refuse a capture of `activities` this session does not collect on this device."""
        if not self.active or not self.collection.features & Feature.ACTIVITY:
            raise RuntimeError("the active Profiler did not request native activity collection")
        if activities & ~self.collection.activities:
            raise RuntimeError("the active Profiler did not enable the requested activity kinds")
        if self.collector.device_index != self.collection.device_index:
            raise RuntimeError("selected device differs from the actual current CUDA device")

    def _demands_activity(self) -> bool:
        """Whether this session named ACTIVITY, which a GPU-less host cannot serve.

        Under DEFAULT such a host should still profile its Python.
        """
        wanted = self.collection.features
        return bool(wanted & self.Feature.ACTIVITY) and wanted != self.Feature.DEFAULT

    def _evidence(self, *, observed: bool) -> DeviceEvidence:
        """Say whether device evidence came back, or was never asked for in the first place."""
        if observed:
            return DeviceEvidence.COLLECTED
        wanted = self.collection.features & (self.Feature.DEVICE | self.Feature.ACTIVITY)
        return DeviceEvidence.ABSENT if wanted else DeviceEvidence.UNSOUGHT

    def _retrieve(
        self, *, since: int, until: int, activities: NativeActivity
    ) -> tuple[tuple[KernelTrace, ...], tuple[MemcpyTrace, ...]]:
        """Return the interval's kernels and copies, keeping only the kinds `activities` names."""
        kernels = tuple(self.collector.kernels(since=since, until=until))
        memcpys = tuple(self.collector.memcpys(since=since, until=until))
        return (
            kernels if activities & NativeActivity.KERNEL else (),
            memcpys if activities & NativeActivity.MEMCPY else (),
        )

    def _run_checkpointed[Answer](
        self, work: Callable[[], Answer], activities: NativeActivity
    ) -> tuple[Answer, int, int]:
        """Run `work`, then checkpoint the shared stream even when it raises."""
        try:
            answer = work()
        finally:
            until, loss_after = self.collector.checkpoint(activities)
        return answer, until, loss_after

    def _selected(self) -> GPU | None:
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

    def _stop_collector(self) -> None:
        """Stop the native collector, when this session collected device activity."""
        if self.collection.features & self.Feature.ACTIVITY and self.sampler.gpu is not None:
            self.collector.stop()

    def capture_window[Answer](
        self, work: Callable[[], Answer], activities: NativeActivity
    ) -> tuple[Answer, Profile]:
        """Checkpoint the shared stream, run `work` once, then retrieve exactly that interval.

        The profiler must be entered; `Profiler.capture` opens one when none is active.
        """
        self._capturable(activities)
        since, loss_before = self.collector.checkpoint(activities)
        start_ns = self.spans.tracer.timestamp()
        began = time.perf_counter_ns()
        answer, until, loss_after = self._run_checkpointed(work, activities)
        wall_ns = time.perf_counter_ns() - began
        end_ns = self.spans.tracer.timestamp()
        if loss_after != loss_before:
            raise RuntimeError(f"native activity window lost {loss_after - loss_before} records")
        kernels, memcpys = self._retrieve(since=since, until=until, activities=activities)
        return answer, Profile(
            device=self.sampler.gpu.label if self.sampler.gpu is not None else "",
            device_evidence=DeviceEvidence.COLLECTED
            if kernels or memcpys
            else DeviceEvidence.ABSENT,
            kernels=kernels,
            memcpys=memcpys,
            windows=(
                RegionWindow(
                    name=getattr(work, "__qualname__", type(work).__qualname__),
                    start_ns=start_ns,
                    end_ns=end_ns,
                    wall_ns=wall_ns,
                ),
            ),
        )

    def enter(self, name: str) -> int:
        """Open one span in `spans`, as every active `span` asks through this session."""
        return self.spans.enter(name)

    def exit(self, token: int, *, wall_ns: int) -> None:
        """Close the span `token` names in `spans`."""
        self.spans.exit(token, wall_ns=wall_ns)

    def result(self) -> Profile:
        """Freeze the evidence collected so far into one `Profile`."""
        kernels = tuple(self.collector.kernels())
        memcpys = tuple(self.collector.memcpys())
        activities = tuple(self.collector.activities())
        used_gpu = self.sampler.evidence or bool(kernels or memcpys or activities)
        return Profile(
            device=self.sampler.label
            if self.sampler.evidence
            else (self.sampler.gpu.label if used_gpu and self.sampler.gpu is not None else ""),
            device_evidence=self._evidence(observed=used_gpu),
            summaries=tuple(
                RegionSummary.from_snaps(item.name, item.wall_ms, item.samples)
                for item in self.spans.measurements
            ),
            windows=tuple(self.spans.windows),
            kernels=kernels,
            memcpys=memcpys,
            activities=activities,
            dropped_spans=self.spans.dropped,
            dropped_activities=self.collector.dropped(),
        )
