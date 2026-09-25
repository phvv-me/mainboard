# The NVTX and CUPTI backend against a fake `cupti.cupti`: kinds enable in memory, buffers arrive
# synchronously, and the device barrier and context are stubs, so no CUDA initialization is needed.

import types
from collections.abc import Callable, Sequence
from ctypes import c_size_t
from dataclasses import replace
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


_MEMCPY = nv.RawMemcpy(kind="HtoD", start_ns=0, end_ns=1, bytes_moved=1, correlation_id=0)


def test_supported_drops_kinds_that_raise_not_implemented_and_caches_the_rest(
    fake_cupti: FakeCupti,
) -> None:
    fake_cupti.unsupported = (FakeActivityKind.MEMORY,)
    supported = nv.NvtxTracer().supported()
    assert Activity.KERNEL in supported
    assert Activity.MEMORY not in supported  # GB10-style: MEMORY unavailable
    assert nv.NvtxTracer._supported() is supported  # noqa: SLF001  reason=asserts the module-private cache since=2026-08-16


def test_collector_lifecycle_collects_routes_and_then_drops_its_records(
    fake_cupti: FakeCupti,
) -> None:
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
    """An own name wins, a bare callback id is looked up, and neither falls back to the kind."""
    with nv.CuptiCollector(Activity.RUNTIME) as collector:
        assert fake_cupti.completed is not None
        fake_cupti.completed([_runtime_activity(cbid=7)])
        collector.flush()
        record = collector.activities()[0]
        assert record.kind == "runtime"
        assert record.name == "cb_7"  # resolved via cbid since the activity had no name

    anonymous = nv.RawGeneric(
        kind_id=FakeActivityKind.MEMSET,
        kind="memset",
        name=None,
        cbid=None,
        start_ns=0,
        end_ns=1,
        correlation_id=0,
    )
    named = replace(anonymous, name="explicit", cbid=1)
    assert nv.CuptiCollector.activity_name(named) == "explicit"
    assert nv.CuptiCollector.activity_name(anonymous) == "memset"


def test_the_buffer_callbacks_offer_a_sized_buffer_and_tolerate_no_collector(
    fake_cupti: FakeCupti,
) -> None:
    del fake_cupti
    size, count = nv._on_buffer_requested()  # noqa: SLF001  reason=unit-tests the CUPTI buffer-size callback since=2026-08-16
    assert size > 0 and count == 0
    nv._on_buffer_completed([_kernel_activity()])  # noqa: SLF001  reason=no active collector -> no error since=2026-08-16


def test_nested_collection_is_rejected(fake_cupti: FakeCupti) -> None:
    del fake_cupti
    with (
        nv.CuptiCollector(Activity.KERNEL),
        pytest.raises(RuntimeError, match="single-subscriber"),
    ):
        nv.CuptiCollector(Activity.KERNEL).__enter__()


@pytest.mark.parametrize(
    "failure",
    [RuntimeError("initial flush failed"), NotImplementedError()],
    ids=["failed-flush", "partial-enable"],
)
def test_a_failed_start_disables_every_enabled_kind(
    fake_cupti: FakeCupti, monkeypatch: pytest.MonkeyPatch, failure: Exception
) -> None:
    if isinstance(failure, NotImplementedError):
        fake_cupti.unsupported = (FakeActivityKind.MEMCPY,)
    else:
        monkeypatch.setattr(fake_cupti, "activity_flush_all", Mock(side_effect=failure))
    collector = nv.CuptiCollector(Activity.DEFAULT)
    with pytest.raises(type(failure)):
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

    monkeypatch.setattr(fake_cupti, "activity_disable", fail_disable)
    collector = nv.CuptiCollector(Activity.DEFAULT | Activity.MEMSET)
    during_start = phase == "start"
    if during_start:
        fake_cupti.unsupported = (FakeActivityKind.MEMSET,)
    with pytest.raises(RuntimeError) as caught, collector:
        if phase == "work-and-flush":
            monkeypatch.setattr(
                fake_cupti, "activity_flush_all", Mock(side_effect=OSError("flush failed"))
            )
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
    """A never-entered collector pops nothing, never taking somebody else's slot."""
    collector = nv.CuptiCollector(Activity.KERNEL)
    collector.enabled_kinds = (FakeActivityKind.CONCURRENT_KERNEL,)
    collector.running = True
    fake_cupti.enabled.add(FakeActivityKind.CONCURRENT_KERNEL)
    collector.stop()
    assert collector.running is False
    assert fake_cupti.enabled == set()

    nv.CuptiCollector(Activity.KERNEL).stop()  # never entered, so not running
    assert nv._active == []  # noqa: SLF001  reason=asserts the module-private active-collector stack since=2026-08-16


def test_annotation_goes_to_nvtx_while_the_deep_trace_goes_to_cupti(
    fake_cupti: FakeCupti, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each `start` closer ends the range it opened, not whichever is on top."""
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
    """An unknown domain is skipped, and a second stop has no subscriber left to release."""
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


def test_device_sync_requires_the_runtime_binding(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(nv, "_runtime_loaded", True)
    monkeypatch.setattr(nv, "cuda_runtime", None)
    with pytest.raises(RuntimeError, match="runtime binding is required"):
        nv._sync()


@pytest.mark.parametrize(
    "failure", [None, ImportError, OSError], ids=["loads", "absent", "broken"]
)
def test_the_runtime_binding_is_imported_once_on_first_use(
    monkeypatch: pytest.MonkeyPatch, failure: type[Exception] | None
) -> None:
    """A missing or broken CUDA library answers no binding, and neither outcome is retried."""
    binding = types.SimpleNamespace()
    imports: list[str] = []

    def load(name: str) -> types.SimpleNamespace:
        imports.append(name)
        if failure is not None:
            raise failure(name)
        return binding

    monkeypatch.setattr(nv, "_runtime_loaded", False)
    monkeypatch.setattr(nv, "cuda_runtime", None)
    monkeypatch.setattr(nv, "import_module", load)
    expected = binding if failure is None else None
    assert nv._runtime() is expected
    assert nv._runtime() is expected
    assert imports == ["cuda.bindings.runtime"]


def install_scope(
    monkeypatch: pytest.MonkeyPatch, device: tuple[int, int] | None, context: tuple[int, int]
) -> None:
    """Stub `cudaGetDevice` as `device` (None: no runtime), `cuCtxGetCurrent` as `context`."""
    runtime = None if device is None else types.SimpleNamespace(cudaGetDevice=lambda: device)
    driver = types.SimpleNamespace(cuCtxGetCurrent=lambda: context)
    monkeypatch.setattr(nv, "_runtime_loaded", True)
    monkeypatch.setattr(nv, "cuda_runtime", runtime)
    monkeypatch.setattr(nv, "cupti", FakeCupti())
    monkeypatch.setattr(nv, "import_module", {"cuda.bindings.driver": driver}.__getitem__)


def test_the_window_scope_is_the_visible_ordinal_and_the_cupti_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_scope(monkeypatch, device=(0, 3), context=(0, 0xC0))
    assert nv.CuptiCollector._scope() == (3, 1, 2)


@pytest.mark.parametrize(
    ("device", "context", "message"),
    [
        (None, (0, 0xC0), "CUDA runtime is required"),
        ((100, 0), (0, 0xC0), "cudaGetDevice failed with CUDA status 100"),
        ((0, 3), (201, 0xC0), "cuCtxGetCurrent failed"),
        ((0, 3), (0, 0), "returned no context"),
    ],
    ids=["no_runtime", "no_device", "driver_refuses", "no_context"],
)
def test_the_window_scope_is_never_substituted_when_unidentified(
    monkeypatch: pytest.MonkeyPatch,
    device: tuple[int, int] | None,
    context: tuple[int, int],
    message: str,
) -> None:
    install_scope(monkeypatch, device, context)
    with pytest.raises(RuntimeError, match=message):
        nv.CuptiCollector._scope()


def test_an_idle_collector_has_no_window_cursor_and_no_device() -> None:
    collector = nv.CuptiCollector(Activity.KERNEL)
    with pytest.raises(RuntimeError, match="running collector"):
        collector.checkpoint(Activity.KERNEL)
    with pytest.raises(RuntimeError, match="no current CUDA context"):
        _ = collector.device_index


@pytest.mark.parametrize("status", [0, 1, 999])
def test_device_sync_checks_the_return_status(
    monkeypatch: pytest.MonkeyPatch, status: int
) -> None:
    """Only cudaSuccess completes the barrier; launch errors remain failures."""
    monkeypatch.setattr(nv, "_runtime_loaded", True)
    synchronize = Mock(return_value=(status,))
    monkeypatch.setattr(
        nv, "cuda_runtime", types.SimpleNamespace(cudaDeviceSynchronize=synchronize)
    )
    if status:
        with pytest.raises(RuntimeError, match=f"CUDA status {status}"):
            nv._sync()
    else:
        nv._sync()
    synchronize.assert_called_once_with()


def test_native_loss_is_added_once_even_without_delivered_records(fake_cupti: FakeCupti) -> None:
    """Reset-on-read native loss is counted apart from records the bounded buffer overwrote."""
    fake_cupti.native_dropped = 17  # stale activity before this capture is excluded
    with nv.CuptiCollector(Activity.KERNEL, max_records=1) as collector:
        assert collector.dropped() == 0
        fake_cupti.native_dropped = 3
        collector.flush()  # no completion callback: all records may have been lost
        assert collector.native_dropped_records == 3
        assert collector.dropped_records == 0
        collector.flush()
        assert collector.dropped() == 3
        collector.append(_MEMCPY)
        collector.append(_MEMCPY)
        assert len(collector.records) == 1
        fake_cupti.native_dropped = 2
    assert collector.native_dropped_records == 5
    assert collector.dropped_records == 1
    assert collector.dropped() == 6
    assert fake_cupti.drop_queries == [(0, 0)] * 4


def test_reset_discards_both_loss_counts_at_the_same_boundary(fake_cupti: FakeCupti) -> None:
    """Reset drains pending native loss before clearing the measurement window."""
    with nv.CuptiCollector(Activity.KERNEL, max_records=1) as collector:
        collector.append(_MEMCPY)
        collector.append(_MEMCPY)
        fake_cupti.native_dropped = 3
        collector.reset()
        assert collector.dropped() == 0
        assert collector.memcpys() == []
        fake_cupti.native_dropped = 2
    assert collector.dropped() == 2


@pytest.mark.parametrize("phase", ["start", "flush", "stop"])
@pytest.mark.parametrize(
    ("field", "value", "message"),
    [("sync_status", 999, "CUDA status 999"), ("drop_failure", True, "native loss query failed")],
)
def test_failed_capture_barrier_or_loss_query_refuses_and_cleans_up(
    fake_cupti: FakeCupti, phase: str, field: str, value: int, message: str
) -> None:
    """Neither query failure nor asynchronous CUDA failure becomes zero reported loss."""
    collector = nv.CuptiCollector(Activity.KERNEL)
    if phase != "start":
        collector.__enter__()
    setattr(fake_cupti, field, value)
    operation = {"start": collector.__enter__, "flush": collector.flush, "stop": collector.stop}
    with pytest.raises(RuntimeError, match=message):
        operation[phase]()
    if phase == "flush":
        with pytest.raises(RuntimeError, match=message):
            collector.stop()
    assert fake_cupti.enabled == set()
    assert collector.running is False
    assert nv._active == []
