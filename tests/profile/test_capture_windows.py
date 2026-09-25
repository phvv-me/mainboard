"""Software-only controls for the unapplied shared-owner proposal."""

from collections.abc import Iterator
from types import SimpleNamespace

import pytest

from mainboard.probe import GPU
from mainboard.profile import Activity, Profiler
from mainboard.profile.providers.nvidia import tracer
from mainboard.profile.providers.nvidia.tracer import CuptiCollector, RawGeneric, RawKernel
from mainboard.profile.result import DeviceEvidence
from mainboard.profile.spans import activate, active, deactivate

from .support import FakeActivityKind, FakeGPU, RecordingSession


@pytest.fixture
def owner(monkeypatch: pytest.MonkeyPatch) -> Iterator[Profiler]:
    """Use only a raw in-memory collector, never CUDA runtime initialization.

    A window reads the enabled kinds back through CUPTI's enum, so the module stands in for the
    CUPTI a GPU-less host does not install.
    """
    monkeypatch.setattr(tracer, "cupti", SimpleNamespace(ActivityKind=FakeActivityKind))
    session = Profiler(features=Profiler.Feature.ACTIVITY, activities=Activity.DEFAULT)
    collector = CuptiCollector()
    collector.running = True
    collector.enabled_kinds = (10, 1)
    collector.scope = (0, 1, 2)
    monkeypatch.setattr(collector, "flush", lambda: None)
    session.collector = collector
    session.active = True
    activate(session)
    try:
        yield session
    finally:
        deactivate(session)
        session.active = False
        collector.running = False


def append(owner: Profiler, name: str, *, context: int | None = 2, stream: int | None = 7) -> None:
    """Deliver one typed native record to the in-memory owner."""
    assert isinstance(owner.collector, CuptiCollector)
    owner.collector.append(
        RawKernel(
            name=name,
            start_ns=10,
            end_ns=20,
            grid="1x1x1",
            block="32x1x1",
            static_shared_mem=0,
            dynamic_shared_mem=0,
            registers=1,
            correlation_id=1,
            device_id=1,
            context_id=context,
            stream_id=stream,
        )
    )


def test_nested_views_do_not_duplicate_physical_records(owner: Profiler) -> None:
    views = []

    def inside() -> int:
        append(owner, "fresh-gemm")
        return 23

    def outside() -> int:
        append(owner, "before")
        answer, view = Profiler.capture(inside, activities=Activity.KERNEL)
        views.append(view)
        append(owner, "after")
        return answer

    answer, outer = Profiler.capture(outside)
    assert answer == 23
    assert [k.name for k in views[0].kernels] == ["fresh-gemm"]
    assert [k.name for k in outer.kernels] == ["before", "fresh-gemm", "after"]
    assert [k.name for k in owner.result().kernels] == ["before", "fresh-gemm", "after"]


def test_each_window_observes_the_current_operation(owner: Profiler) -> None:
    _, first = Profiler.capture(lambda: append(owner, "kernel-A"))
    _, second = Profiler.capture(lambda: append(owner, "kernel-B"))
    assert [k.name for k in first.kernels] == ["kernel-A"]
    assert [k.name for k in second.kernels] == ["kernel-B"]


def test_empty_is_absent_not_positive_evidence(owner: Profiler) -> None:
    _, view = Profiler.capture(lambda: None)
    assert not view.kernels
    assert view.device_evidence is DeviceEvidence.ABSENT


@pytest.mark.parametrize("context", [None, 999])
def test_unknown_or_foreign_context_refuses(owner: Profiler, context: int | None) -> None:
    with pytest.raises(RuntimeError, match="foreign or unknown"):
        Profiler.capture(lambda: append(owner, "foreign", context=context))


def test_native_loss_refuses_even_without_delivered_records(owner: Profiler) -> None:
    collector = owner.collector

    def lose() -> None:
        assert isinstance(collector, CuptiCollector)
        collector.lost_records += 5
        collector.native_dropped_records += 5

    with pytest.raises(RuntimeError, match="lost 5 records"):
        Profiler.capture(lose)
    assert owner.result().dropped_activities == 5


def test_overwrite_refuses(owner: Profiler) -> None:
    collector = owner.collector
    assert isinstance(collector, CuptiCollector)
    collector.records = collector.records.__class__(maxlen=1)

    def overwrite() -> None:
        append(owner, "first")
        append(owner, "second")

    with pytest.raises(RuntimeError, match="lost 1 records"):
        Profiler.capture(overwrite)


def test_reset_cannot_hide_lost_scope_records(owner: Profiler) -> None:
    def reset() -> None:
        append(owner, "discarded")
        owner.collector.reset()
        append(owner, "after-reset")

    with pytest.raises(RuntimeError, match="reset or overwritten"):
        Profiler.capture(reset)


def test_work_failure_is_not_a_completed_profile(owner: Profiler) -> None:
    def fail() -> None:
        append(owner, "before-error")
        raise ValueError("intentional work error")

    with pytest.raises(ValueError, match="intentional work error"):
        Profiler.capture(fail)
    assert len(owner.result().kernels) == 1


def test_device_mismatch_refuses_before_work(owner: Profiler) -> None:
    with pytest.raises(RuntimeError, match="different CUDA-visible"):
        Profiler.capture(lambda: pytest.fail("must not execute"), device_index=1)


def test_host_only_owner_is_not_silently_upgraded(owner: Profiler) -> None:
    owner.collection = owner.collection.model_copy(update={"features": Profiler.Feature.SPANS})
    with pytest.raises(RuntimeError, match="did not request"):
        Profiler.capture(lambda: pytest.fail("must not execute"))


def test_disabled_activity_kind_refuses(owner: Profiler) -> None:
    owner.collection = owner.collection.model_copy(update={"activities": Activity.KERNEL})
    with pytest.raises(RuntimeError, match="did not enable"):
        Profiler.capture(lambda: pytest.fail("must not execute"))


@pytest.mark.parametrize("declared", [Activity.DEFAULT, Activity.ALL])
def test_actual_enabled_kinds_override_declared_policy(
    owner: Profiler, declared: Activity
) -> None:
    owner.collection = owner.collection.model_copy(update={"activities": declared})
    collector = owner.collector
    assert isinstance(collector, CuptiCollector)
    collector.enabled_kinds = (10,)
    with pytest.raises(RuntimeError, match="not actually enabled"):
        Profiler.capture(lambda: pytest.fail("must not execute"), activities=Activity.MEMCPY)
    _, capture = Profiler.capture(
        lambda: append(owner, "enabled-kernel"), activities=Activity.KERNEL
    )
    assert [kernel.name for kernel in capture.kernels] == ["enabled-kernel"]


def test_nondefault_stream_is_retained(owner: Profiler) -> None:
    _, view = Profiler.capture(lambda: append(owner, "stream-9", stream=9))
    assert view.kernels[0].stream_id == 9


def test_unknown_stream_refuses(owner: Profiler) -> None:
    with pytest.raises(RuntimeError, match="incomplete native timing or stream"):
        Profiler.capture(lambda: append(owner, "unknown-stream", stream=None))


def test_foreign_thread_refuses_before_synchronization(owner: Profiler) -> None:
    collector = owner.collector
    assert isinstance(collector, CuptiCollector)
    collector.owner_thread = -1
    with pytest.raises(RuntimeError, match="issuing thread or CUDA context"):
        CuptiCollector.flush(collector)


def test_changed_context_refuses_before_synchronization(
    owner: Profiler, monkeypatch: pytest.MonkeyPatch
) -> None:
    collector = owner.collector
    assert isinstance(collector, CuptiCollector)
    monkeypatch.setattr(collector, "_scope", lambda: (1, 55, 99))
    with pytest.raises(RuntimeError, match="issuing thread or CUDA context"):
        CuptiCollector.flush(collector)


@pytest.mark.parametrize("activities", [Activity(0), Activity.RUNTIME], ids=["none", "runtime"])
def test_a_window_collects_only_kernels_and_copies(activities: Activity) -> None:
    """Asking for nothing, or for a kind a window cannot bound, is refused before any owner."""
    with pytest.raises(ValueError, match="kernels and memory copies"):
        Profiler.capture(lambda: pytest.fail("must not execute"), activities=activities)


def test_a_span_owner_that_is_not_a_profiler_cannot_open_a_window() -> None:
    """A bare span session has no native collector, so a window never borrows its slot."""
    session = RecordingSession()
    activate(session)
    try:
        with pytest.raises(RuntimeError, match="cannot provide native activity windows"):
            Profiler.capture(lambda: pytest.fail("must not execute"))
    finally:
        deactivate(session)


def test_without_an_owner_a_window_opens_its_own_session_and_closes_it(
    one_gpu: FakeGPU, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The window owns the session it opened, so a backend without windows leaves none behind."""
    monkeypatch.setattr(GPU, "all", staticmethod(lambda: (one_gpu,)))
    with pytest.raises(RuntimeError, match="no synchronized CUDA device"):
        Profiler.capture(lambda: pytest.fail("must not execute"))
    assert active() is None


def test_a_collector_on_another_device_refuses_before_work(owner: Profiler) -> None:
    """The collector's current device, not the declared policy, decides whose work is seen."""
    collector = owner.collector
    assert isinstance(collector, CuptiCollector)
    collector.scope = (1, 1, 2)
    with pytest.raises(RuntimeError, match="differs from the actual current CUDA device"):
        Profiler.capture(lambda: pytest.fail("must not execute"))


def test_a_copy_only_window_leaves_out_the_kernels_it_saw(owner: Profiler) -> None:
    _, view = Profiler.capture(lambda: append(owner, "unwanted"), activities=Activity.MEMCPY)
    assert view.kernels == ()
    assert view.device_evidence is DeviceEvidence.ABSENT


def test_api_records_in_a_window_carry_no_context_and_are_not_refused(owner: Profiler) -> None:
    """A runtime API record has no device or stream, so the scope check passes over it."""
    collector = owner.collector
    assert isinstance(collector, CuptiCollector)

    def launch() -> None:
        collector.append(
            RawGeneric(
                kind_id=FakeActivityKind.RUNTIME,
                kind="runtime",
                name="cudaLaunchKernel",
                cbid=None,
                start_ns=5,
                end_ns=8,
                correlation_id=1,
            )
        )
        append(owner, "launched")

    _, view = Profiler.capture(launch)
    assert [kernel.name for kernel in view.kernels] == ["launched"]
