# The NVIDIA NVTX and CUPTI Activity backend, driven by a fake `cupti` module. CUPTI is
# single-subscriber and GPU-only, so here the whole `cupti.cupti` surface is a fake where activity
# kinds enable and disable in memory, buffers are delivered synchronously, and the device sync is a
# counter. That covers the collector lifecycle, the buffer-routing callback, device-support probing
# (including a kind that raises `NotImplementedError`, like `MEMORY` on GB10) and the callback-API
# call counter, none of it needing real hardware.

import types
from collections.abc import Callable, Sequence
from ctypes import c_size_t
from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest

from mainboard.profile import Activity, DeviceEvidence, Profile, Profiler, TraceCollector, annotate
from mainboard.profile.providers import nvidia_tracer as nv
from mainboard.trials import Log

from ..support import FakeActivityKind, one_process_gpu

if TYPE_CHECKING:
    from mainboard.profile.protocols import RawActivity
    from mainboard.profile.providers.nvidia.protocols import CallbackData, Subscriber


class FakeCallbackDomain:
    RUNTIME_API = "runtime_api"
    DRIVER_API = "driver_api"
    NVTX = "nvtx"


class FakeApiCallbackSite:
    API_ENTER = 1
    API_EXIT = 2


class FakeCupti:
    """In-memory stand-in for the `cupti.cupti` module.

    `unsupported` names the activity kinds whose `activity_enable` raises
    `NotImplementedError`, modelling a device that lacks them (e.g. GB10 + MEMORY).
    """

    ActivityKind = FakeActivityKind
    CallbackDomain = FakeCallbackDomain
    ApiCallbackSite = FakeApiCallbackSite

    def __init__(self, unsupported: Sequence[int] = ()) -> None:
        self.unsupported = unsupported
        self.enabled: set[int] = set()
        self.flushes = 0
        self.native_dropped = 0
        self.drop_queries: list[tuple[int, int]] = []
        self.sync_status = 0
        self.drop_failure = False
        self.completed: Callable[[list[RawActivity]], None] | None = None
        self.subscribed: list[Subscriber] = []
        self.callback: Callable[[None, int, int, CallbackData], None] | None = None

    def activity_disable(self, kind: int) -> None:
        self.enabled.discard(kind)

    def activity_enable(self, kind: int) -> None:
        if kind in self.unsupported:
            raise NotImplementedError(kind)
        self.enabled.add(kind)

    def activity_flush_all(self, _flag: int) -> None:
        self.flushes += 1

    def activity_get_num_dropped_records(self, context: int, stream_id: int, dropped: int) -> None:
        if self.drop_failure:
            raise RuntimeError("native loss query failed")
        self.drop_queries.append((context, stream_id))
        c_size_t.from_address(dropped).value = self.native_dropped
        self.native_dropped = 0

    def activity_register_callbacks(
        self,
        requested: Callable[[], tuple[int, int]],
        completed: Callable[[list[RawActivity]], None],
    ) -> None:
        del requested
        self.completed = completed

    def enable_domain(self, _on: int, _sub: Subscriber, _domain: int) -> None:
        pass

    def get_callback_name(self, _domain: int, cbid: int) -> str:
        return f"cb_{cbid}"

    def get_context_id(self, context: int) -> int:
        return 2

    def get_device_id(self, context: int) -> int:
        return 1

    def get_timestamp(self) -> int:
        return 123

    def subscribe(
        self, callback: Callable[[None, int, int, CallbackData], None], _userdata: None
    ) -> Subscriber:
        self.callback = callback
        token = object()
        self.subscribed.append(token)
        return token

    def unsubscribe(self, token: Subscriber) -> None:
        self.subscribed.remove(token)


@pytest.fixture
def fake_cupti(monkeypatch: pytest.MonkeyPatch) -> FakeCupti:
    """Install a fresh fake CUPTI and reset the module's global subscriber state."""
    cupti = FakeCupti()
    monkeypatch.setattr(nv, "cupti", cupti)
    monkeypatch.setattr(
        nv,
        "cuda_runtime",
        types.SimpleNamespace(cudaDeviceSynchronize=lambda: (cupti.sync_status,)),
    )
    monkeypatch.setattr(nv, "_runtime_loaded", True)
    monkeypatch.setattr(nv.CuptiCollector, "_scope", staticmethod(lambda: (0, 1, 2)))
    monkeypatch.setattr(nv, "_active", [])
    monkeypatch.setattr(nv, "_registered", False)
    monkeypatch.setattr(nv, "_supported_kinds", None)
    monkeypatch.setattr(nv, "_label", {})
    monkeypatch.setattr(nv, "_domain", {})
    return cupti


def _kernel_activity(name: str = "gemm") -> RawActivity:
    return types.SimpleNamespace(
        kind=FakeActivityKind.CONCURRENT_KERNEL,
        name=name,
        start=0,
        end=1000,
        grid_x=1,
        grid_y=1,
        grid_z=1,
        block_x=128,
        block_y=1,
        block_z=1,
        static_shared_memory=0,
        dynamic_shared_memory=0,
        registers_per_thread=32,
    )


def _memcpy_activity() -> RawActivity:
    return types.SimpleNamespace(
        kind=FakeActivityKind.MEMCPY, copy_kind=1, start=0, end=500, bytes=2048
    )


def _runtime_activity(cbid: int = 7) -> RawActivity:
    return types.SimpleNamespace(
        kind=FakeActivityKind.RUNTIME, name=None, cbid=cbid, start=0, end=10, correlation_id=99
    )


def test_supported_drops_kinds_that_raise_not_implemented_and_caches_the_rest(
    fake_cupti: FakeCupti,
) -> None:
    """A kind whose `activity_enable` raises is excluded, and the probe runs only once."""
    fake_cupti.unsupported = (FakeActivityKind.MEMORY,)
    supported = nv.NvtxTracer().supported()
    assert Activity.KERNEL in supported
    assert Activity.MEMORY not in supported  # GB10-style: MEMORY unavailable
    assert nv.NvtxTracer._supported() is supported  # noqa: SLF001  reason=asserts the module-private cache since=2026-08-16


def test_collector_lifecycle_collects_routes_and_then_drops_its_records(
    fake_cupti: FakeCupti,
) -> None:
    """A collector enables kinds, a completed buffer routes typed records, `reset` clears them.

    A kind that was never enabled is ignored by the router, and leaving the context drains
    the buffer and disables every native kind the capture turned on.
    """
    with nv.CuptiCollector(Activity.KERNEL | Activity.MEMCPY) as collector:
        assert fake_cupti.completed is not None
        fake_cupti.completed([_kernel_activity(), _memcpy_activity(), _runtime_activity()])
        collector.flush()
        assert collector.kernels()[0].name == "gemm"
        assert collector.memcpys()[0].bytes_moved == 2048
        # RUNTIME wasn't enabled here, so its record is ignored by the router
        assert collector.activities() == []
        collector.reset()
        assert collector.kernels() == []
    assert fake_cupti.flushes > 0  # stop drained the buffer
    assert fake_cupti.enabled == set()  # every activity kind was disabled at capture end


def test_a_generic_activity_resolves_its_name_after_the_callback_returns(
    fake_cupti: FakeCupti,
) -> None:
    """An enabled non-kernel/memcpy kind becomes a generic record with a resolved name.

    Name resolution is deferred past the buffer callback, so a record that carries its own
    name keeps it, one that carries only a callback id looks the name up, and one with
    neither falls back to its kind label.
    """
    with nv.CuptiCollector(Activity.RUNTIME) as collector:
        assert fake_cupti.completed is not None
        fake_cupti.completed([_runtime_activity(cbid=7)])
        collector.flush()
        record = collector.activities()[0]
        assert record.kind == "runtime"
        assert record.name == "cb_7"  # resolved via cbid since the activity had no name

    named = nv.RawGeneric(
        kind_id=FakeActivityKind.RUNTIME,
        kind="runtime",
        name="explicit",
        cbid=1,
        start_ns=0,
        end_ns=1,
        correlation_id=0,
    )
    anonymous = nv.RawGeneric(
        kind_id=FakeActivityKind.MEMSET,
        kind="memset",
        name=None,
        cbid=None,
        start_ns=0,
        end_ns=1,
        correlation_id=0,
    )
    assert nv.CuptiCollector.activity_name(named) == "explicit"
    assert nv.CuptiCollector.activity_name(anonymous) == "memset"


def test_the_buffer_callbacks_offer_a_sized_buffer_and_tolerate_no_collector(
    fake_cupti: FakeCupti,
) -> None:
    """A buffer completed with nobody listening is dropped.

    CUPTI is handed a sized, empty buffer, and the drop never raises.
    """
    del fake_cupti
    size, count = nv._on_buffer_requested()  # noqa: SLF001  reason=unit-tests the CUPTI buffer-size callback since=2026-08-16
    assert size > 0 and count == 0
    nv._on_buffer_completed([_kernel_activity()])  # noqa: SLF001  reason=no active collector -> no error since=2026-08-16


def test_nested_collection_is_rejected(fake_cupti: FakeCupti) -> None:
    """CUPTI is single-subscriber, so a second simultaneous collector is refused."""
    del fake_cupti
    with (
        nv.CuptiCollector(Activity.KERNEL),
        pytest.raises(RuntimeError, match="single-subscriber"),
    ):
        nv.CuptiCollector(Activity.KERNEL).__enter__()


def test_failed_collector_start_disables_every_enabled_kind(
    fake_cupti: FakeCupti, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed initial flush must not leave process-wide CUPTI activity enabled."""

    def fail(_flag: int) -> None:
        raise RuntimeError("flush failed")

    monkeypatch.setattr(fake_cupti, "activity_flush_all", fail)
    collector = nv.CuptiCollector(Activity.KERNEL)
    with pytest.raises(RuntimeError, match="flush failed"):
        collector.__enter__()
    assert collector.enabled_kinds == ()
    assert fake_cupti.enabled == set()


def test_partial_enable_is_rolled_back(fake_cupti: FakeCupti) -> None:
    fake_cupti.unsupported = (FakeActivityKind.MEMCPY,)
    collector = nv.CuptiCollector(Activity.DEFAULT)
    with pytest.raises(NotImplementedError):
        collector.__enter__()
    assert fake_cupti.enabled == set()
    assert collector.enabled_kinds == ()
    assert not collector.running


@pytest.mark.parametrize("phase", ["start", "work", "work-and-flush"])
def test_every_disable_is_attempted_and_every_error_retained(
    fake_cupti: FakeCupti, monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    attempted = []

    def fail_disable(kind: int) -> None:
        attempted.append(kind)
        raise RuntimeError(f"disable {kind}")

    def fail_flush(_flag: int) -> None:
        raise OSError("flush failed")

    monkeypatch.setattr(fake_cupti, "activity_disable", fail_disable)
    collector = nv.CuptiCollector(Activity.DEFAULT | Activity.MEMSET)
    during_start = phase == "start"
    if during_start:
        fake_cupti.unsupported = (FakeActivityKind.MEMSET,)
    with pytest.raises(RuntimeError) as caught, collector:
        if phase == "work-and-flush":
            monkeypatch.setattr(fake_cupti, "activity_flush_all", fail_flush)
        raise ValueError("work failed")
    assert attempted == (
        [FakeActivityKind.MEMCPY, FakeActivityKind.CONCURRENT_KERNEL]
        if during_start
        else [FakeActivityKind.MEMSET, FakeActivityKind.MEMCPY, FakeActivityKind.CONCURRENT_KERNEL]
    )
    errors: list[BaseException] = []
    current: BaseException | None = caught.value
    while current is not None:
        errors.append(current)
        current = current.__context__
    assert [str(error) for error in errors[: len(attempted)]] == [
        f"disable {kind}" for kind in reversed(attempted)
    ]
    assert isinstance(errors[-1], NotImplementedError if during_start else ValueError)
    if phase == "work-and-flush":
        assert isinstance(errors[-2], OSError)
    assert len(errors) == len(attempted) + 1 + (phase == "work-and-flush")
    assert collector.enabled_kinds == ()
    assert not collector.running


def test_callback_failure_remains_tainted_after_reset(fake_cupti: FakeCupti) -> None:
    with (
        pytest.raises(RuntimeError, match="tainted"),
        nv.CuptiCollector(Activity.KERNEL) as collector,
    ):
        assert fake_cupti.completed is not None
        malformed = types.SimpleNamespace(kind=FakeActivityKind.CONCURRENT_KERNEL)
        with pytest.raises(AttributeError):
            fake_cupti.completed([_kernel_activity("retained-prefix"), malformed])
        assert [kernel.name for kernel in collector.kernels()] == ["retained-prefix"]
        with pytest.raises(RuntimeError, match="tainted"):
            collector.checkpoint(Activity.KERNEL)
        with pytest.raises(RuntimeError, match="tainted"):
            collector.reset()
        fake_cupti.completed([_kernel_activity("later-record")])
        assert collector.callback_failed
        assert [kernel.name for kernel in collector.kernels()] == [
            "retained-prefix",
            "later-record",
        ]
        with pytest.raises(RuntimeError, match="tainted"):
            collector.checkpoint(Activity.KERNEL)
    assert not collector.running
    assert collector.enabled_kinds == ()


def test_callback_failure_during_flush_refuses_checkpoint(
    fake_cupti: FakeCupti, monkeypatch: pytest.MonkeyPatch
) -> None:
    with (
        pytest.raises(RuntimeError, match="tainted"),
        nv.CuptiCollector(Activity.KERNEL) as collector,
    ):

        def malformed_callback(_flag: int) -> None:
            assert fake_cupti.completed is not None
            with pytest.raises(AttributeError):
                fake_cupti.completed([types.SimpleNamespace(kind=FakeActivityKind.MEMCPY)])

        monkeypatch.setattr(fake_cupti, "activity_flush_all", malformed_callback)
        with pytest.raises(RuntimeError, match="tainted"):
            collector.checkpoint(Activity.KERNEL)


@pytest.mark.parametrize("logged", [False, True])
def test_ordinary_owner_refuses_taint_but_retains_partial_evidence(
    fake_cupti: FakeCupti, monkeypatch: pytest.MonkeyPatch, logged: bool
) -> None:
    monkeypatch.setattr(annotate, "_tracer", nv.NvtxTracer())
    owner = Profiler(gpus=(one_process_gpu(),), features=Profiler.Feature.ACTIVITY)
    log = Log.__new__(Log)
    artifact, event = Mock(), Mock()
    monkeypatch.setattr(log, "artifact", artifact)
    monkeypatch.setattr(log, "_event", event)
    monkeypatch.setattr(Profiler, "under", classmethod(lambda cls, policy: owner))
    context = log.profile(name="partial", collection=owner.collection) if logged else owner
    with pytest.raises(RuntimeError, match="tainted"), context:
        assert fake_cupti.completed is not None
        with pytest.raises(AttributeError):
            fake_cupti.completed(
                [
                    _kernel_activity("retained-prefix"),
                    types.SimpleNamespace(kind=FakeActivityKind.CONCURRENT_KERNEL),
                ]
            )
    assert not owner.active
    assert fake_cupti.enabled == set()
    snapshot = owner.result()
    assert [kernel.name for kernel in snapshot.kernels] == ["retained-prefix"]
    assert snapshot.dropped_activities == 0  # unknown conversion loss is not a numeric count
    assert snapshot.device_evidence is DeviceEvidence.COLLECTED  # observed, not complete
    if logged:
        artifact.assert_called_once()
        saved = Profile.model_validate_json(artifact.call_args.args[0])
        assert saved == snapshot
        assert event.call_args.args[1]["completed"] is False


def test_stop_cleans_up_even_when_the_active_slot_was_lost(fake_cupti: FakeCupti) -> None:
    """Cleanup disables native kinds even if external state lost the active slot.

    A collector that was never entered is not on the stack at all, so stopping it pops
    nothing rather than taking somebody else's slot.
    """
    collector = nv.CuptiCollector(Activity.KERNEL)
    collector.enabled_kinds = (FakeActivityKind.CONCURRENT_KERNEL,)
    collector.running = True
    fake_cupti.enabled.add(FakeActivityKind.CONCURRENT_KERNEL)
    collector.stop()
    assert collector.running is False
    assert fake_cupti.enabled == set()

    nv.CuptiCollector(Activity.KERNEL).stop()  # never entered, so not running
    assert nv._active == []  # noqa: SLF001  reason=asserts the module-private active-collector stack since=2026-08-16


def test_raw_activity_buffer_is_bounded() -> None:
    """Past its record cap the capture buffer overwrites the oldest and counts the loss."""
    collector = nv.CuptiCollector(max_records=1)
    record = nv.RawMemcpy(copy_kind=1, start_ns=0, end_ns=1, bytes_moved=1, correlation_id=0)
    collector.append(record)
    collector.append(record)
    assert len(collector.records) == 1
    assert collector.dropped_records == 1
    assert collector.dropped() == 1


def test_annotation_goes_to_nvtx_while_the_deep_trace_goes_to_cupti(
    fake_cupti: FakeCupti, monkeypatch: pytest.MonkeyPatch
) -> None:
    """NVTX carries push/pop/mark, and CUPTI backs the collector, the sessions and the clock.

    `start` opens an overlap-safe process range, so each closer ends the range it opened
    rather than whichever one happens to be on top.
    """
    del fake_cupti
    events: list[tuple[str, str | None | tuple[int, int]]] = []
    fake_nvtx = types.SimpleNamespace(
        push_range=lambda name: events.append(("push", name)),
        pop_range=lambda: events.append(("pop", None)),
        start_range=lambda name: events.append(("start", name)) or (len(events), 0),
        end_range=lambda range_id: events.append(("end", range_id)),
        mark=lambda message: events.append(("mark", message)),
    )
    monkeypatch.setattr(nv, "nvtx", fake_nvtx)
    tracer = nv.NvtxTracer()
    assert nv.NvtxTracer.is_available() is True
    tracer.push("r")
    tracer.mark("m")
    tracer.pop()
    finish_first = tracer.start("first")
    finish_second = tracer.start("second")
    finish_first()
    finish_second()
    assert events == [
        ("push", "r"),
        ("mark", "m"),
        ("pop", None),
        ("start", "first"),
        ("start", "second"),
        ("end", (4, 0)),
        ("end", (5, 0)),
    ]
    assert isinstance(tracer.open(Activity.KERNEL), nv.CuptiCollector)
    assert isinstance(tracer.callbacks(), nv.CuptiCallbackSession)
    assert tracer.timestamp() == 123


def test_nvtx_tracer_degrades_without_libraries(monkeypatch: pytest.MonkeyPatch) -> None:
    """With neither NVTX nor CUPTI, the backend is unavailable and a safe no-op."""
    monkeypatch.setattr(nv, "nvtx", None)
    monkeypatch.setattr(nv, "cupti", None)
    tracer = nv.NvtxTracer()
    assert nv.NvtxTracer.is_available() is False
    tracer.push("x")
    tracer.pop()
    tracer.mark("x")
    tracer.start("x")()
    assert tracer.supported() == Activity(0)
    assert isinstance(tracer.open(Activity.KERNEL), TraceCollector)
    assert tracer.callbacks().counts() == {}
    assert isinstance(tracer.timestamp(), int)


def test_the_callback_session_counts_one_api_call_per_enter(fake_cupti: FakeCupti) -> None:
    """A function name is counted once per ENTER callback and never on the way out.

    A domain the CUPTI enum does not carry is skipped rather than subscribed to, and
    stopping twice is safe since the second stop has no subscriber left to release.
    """
    with nv.CuptiCallbackSession(("runtime", "driver", "bogus")) as session:
        enter = types.SimpleNamespace(
            callback_site=FakeApiCallbackSite.API_ENTER, function_name="cudaMalloc"
        )
        exit_site = types.SimpleNamespace(
            callback_site=FakeApiCallbackSite.API_EXIT, function_name="cudaMalloc"
        )
        assert fake_cupti.callback is not None
        fake_cupti.callback(None, 0, 0, enter)
        fake_cupti.callback(None, 0, 0, enter)
        fake_cupti.callback(None, 0, 0, exit_site)  # EXIT is not counted
    assert session.counts() == {"cudaMalloc": 2}
    assert fake_cupti.subscribed == []  # stop unsubscribed
    session.stop()  # subscriber already cleared -> no-op
