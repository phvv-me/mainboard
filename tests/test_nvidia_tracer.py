"""The NVIDIA NVTX + CUPTI Activity backend, driven by a fake `cupti` module.

CUPTI is single-subscriber and GPU-only, so here the whole `cupti.cupti` surface is a
fake: activity kinds enable/disable in memory, buffers are delivered synchronously, and
the device sync is a counter. This exercises the collector lifecycle, the buffer-routing
callback, device-support probing (including a kind that raises `NotImplementedError`,
like `MEMORY` on GB10), and the callback-API call counter — none needing real hardware.
"""

import types
from typing import Any

import pytest

from mainboard.profiling.trace import Activity, TraceCollector
from mainboard.providers.nvidia import tracer as nv


class FakeActivityKind:
    CONCURRENT_KERNEL = 10
    MEMCPY = 1
    MEMSET = 4
    SYNCHRONIZATION = 8
    OVERHEAD = 16
    MEMORY = 32
    JIT = 64
    RUNTIME = 128
    DRIVER = 256
    MEMORY_POOL = 512


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

    def __init__(self, unsupported: tuple[int, ...] = ()) -> None:
        self.unsupported = unsupported
        self.enabled: set[int] = set()
        self.flushes = 0
        self.completed: Any = None
        self.subscribed: list[Any] = []
        self._cb: Any = None

    def activity_register_callbacks(self, requested: Any, completed: Any) -> None:
        self.completed = completed

    def activity_enable(self, kind: int) -> None:
        if kind in self.unsupported:
            raise NotImplementedError(kind)
        self.enabled.add(kind)

    def activity_disable(self, kind: int) -> None:
        self.enabled.discard(kind)

    def activity_flush_all(self, _flag: int) -> None:
        self.flushes += 1

    def get_callback_name(self, _domain: Any, cbid: int) -> str:
        return f"cb_{cbid}"

    def get_timestamp(self) -> int:
        return 123

    def subscribe(self, callback: Any, _userdata: Any) -> Any:
        self._cb = callback
        token = object()
        self.subscribed.append(token)
        return token

    def unsubscribe(self, token: Any) -> None:
        self.subscribed.remove(token)

    def enable_domain(self, _on: int, _sub: Any, _domain: Any) -> None:
        pass


@pytest.fixture
def fake_cupti(monkeypatch: pytest.MonkeyPatch) -> FakeCupti:
    """Install a fresh fake CUPTI and reset the module's global subscriber state."""
    cupti = FakeCupti()
    monkeypatch.setattr(nv, "cupti", cupti)
    monkeypatch.setattr(nv, "cuda_runtime", None)
    monkeypatch.setattr(nv, "_active", [])
    monkeypatch.setattr(nv, "_registered", False)
    monkeypatch.setattr(nv, "_supported_kinds", None)
    monkeypatch.setattr(nv, "_label", {})
    monkeypatch.setattr(nv, "_domain", {})
    return cupti


def _kernel_activity(name: str = "gemm") -> Any:
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


def _memcpy_activity() -> Any:
    return types.SimpleNamespace(
        kind=FakeActivityKind.MEMCPY, copy_kind=1, start=0, end=500, bytes=2048
    )


def _runtime_activity(cbid: int = 7) -> Any:
    return types.SimpleNamespace(
        kind=FakeActivityKind.RUNTIME, name=None, cbid=cbid, start=0, end=10, correlation_id=99
    )


def test_supported_drops_kinds_that_raise_not_implemented(fake_cupti: FakeCupti) -> None:
    """A kind whose `activity_enable` raises is excluded from the supported set."""
    fake_cupti.unsupported = (FakeActivityKind.MEMORY,)
    tracer = nv.NvtxTracer()
    supported = tracer.supported()
    assert Activity.KERNEL in supported
    assert Activity.MEMORY not in supported  # GB10-style: MEMORY unavailable


def test_supported_is_cached(fake_cupti: FakeCupti) -> None:
    """Device-support probing happens once and is reused."""
    assert nv._supported() == nv._supported()


def test_collector_lifecycle_collects_and_routes_records(fake_cupti: FakeCupti) -> None:
    """A collector enables kinds, and a completed buffer routes typed records to it."""
    with nv.CuptiCollector(Activity.KERNEL | Activity.MEMCPY) as collector:
        fake_cupti.completed([_kernel_activity(), _memcpy_activity(), _runtime_activity()])
        collector.flush()
        assert collector.kernels()[0].name == "gemm"
        assert collector.memcpys()[0].bytes_moved == 2048
        # RUNTIME wasn't enabled here, so its record is ignored by the router
        assert collector.activities() == []
    assert fake_cupti.flushes > 0  # stop drained the buffer
    assert fake_cupti.enabled == set()  # every activity kind was disabled at capture end


def test_collector_routes_generic_activity_when_kind_enabled(fake_cupti: FakeCupti) -> None:
    """An enabled non-kernel/memcpy kind becomes a generic record with a resolved name."""
    with nv.CuptiCollector(Activity.RUNTIME) as collector:
        fake_cupti.completed([_runtime_activity(cbid=7)])
        collector.flush()
        record = collector.activities()[0]
        assert record.kind == "runtime"
        assert record.name == "cb_7"  # resolved via cbid since the activity had no name


def test_buffer_completed_ignores_records_with_no_active_collector(fake_cupti: FakeCupti) -> None:
    """A completed buffer with no live collector is silently dropped."""
    nv._on_buffer_completed([_kernel_activity()])  # no active collector -> no error


def test_nested_collection_is_rejected(fake_cupti: FakeCupti) -> None:
    """CUPTI is single-subscriber, so a second simultaneous collector is refused."""
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


def test_stop_cleans_up_even_when_active_slot_was_lost(fake_cupti: FakeCupti) -> None:
    """Cleanup disables native kinds even if external state lost the active slot."""
    collector = nv.CuptiCollector(Activity.KERNEL)
    collector.enabled_kinds = (FakeActivityKind.CONCURRENT_KERNEL,)
    collector.running = True
    fake_cupti.enabled.add(FakeActivityKind.CONCURRENT_KERNEL)
    collector.stop()
    assert collector.running is False
    assert fake_cupti.enabled == set()


def test_collector_reset_drops_records(fake_cupti: FakeCupti) -> None:
    """`reset` flushes in-flight records then clears everything collected so far."""
    with nv.CuptiCollector(Activity.KERNEL) as collector:
        fake_cupti.completed([_kernel_activity()])
        collector.flush()
        assert collector.kernels()
        collector.reset()
        assert collector.kernels() == []


def test_buffer_requested_returns_size_and_count() -> None:
    """The buffer-request callback offers a sized buffer with no pre-filled count."""
    size, count = nv._on_buffer_requested()
    assert size > 0 and count == 0


def test_deferred_activity_name_prefers_explicit_name(fake_cupti: FakeCupti) -> None:
    """Model conversion resolves names only after the buffer callback returns."""
    record = nv.RawGeneric(
        kind_id=FakeActivityKind.RUNTIME,
        kind="runtime",
        name="explicit",
        cbid=1,
        start_ns=0,
        end_ns=1,
        correlation_id=0,
    )
    assert nv.CuptiCollector.activity_name(record) == "explicit"


def test_deferred_activity_name_falls_back_to_kind(fake_cupti: FakeCupti) -> None:
    record = nv.RawGeneric(
        kind_id=FakeActivityKind.MEMSET,
        kind="memset",
        name=None,
        cbid=None,
        start_ns=0,
        end_ns=1,
        correlation_id=0,
    )
    assert nv.CuptiCollector.activity_name(record) == "memset"


def test_raw_activity_buffer_is_bounded() -> None:
    collector = nv.CuptiCollector(max_records=1)
    record = nv.RawMemcpy(copy_kind=1, start_ns=0, end_ns=1, bytes_moved=1)
    collector.append(record)
    collector.append(record)
    assert len(collector.records) == 1
    assert collector.dropped_records == 1
    assert collector.dropped() == 1


def test_sync_is_a_noop_without_runtime(fake_cupti: FakeCupti) -> None:
    """`_sync` does nothing when the CUDA runtime binding is absent."""
    nv._sync()  # cuda_runtime is None in the fixture -> no error


def test_sync_calls_runtime_when_present(monkeypatch: pytest.MonkeyPatch) -> None:
    """`_sync` synchronizes the device when the runtime binding is present."""
    synced: list[int] = []
    monkeypatch.setattr(
        nv, "cuda_runtime", types.SimpleNamespace(cudaDeviceSynchronize=lambda: synced.append(1))
    )
    nv._sync()
    assert synced == [1]


def test_nvtx_tracer_annotation_forwards_to_nvtx(monkeypatch: pytest.MonkeyPatch) -> None:
    """NVTX push/pop/mark forward to the nvtx library when it is present."""
    events: list[tuple[str, Any]] = []
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


def test_nvtx_tracer_timestamp_uses_cupti(fake_cupti: FakeCupti) -> None:
    """The device clock comes from CUPTI when available."""
    assert nv.NvtxTracer().timestamp() == 123


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


def test_callback_session_counts_api_calls(fake_cupti: FakeCupti) -> None:
    """The callback session counts an API function name once per ENTER callback."""
    with nv.CuptiCallbackSession(("runtime", "driver", "bogus")) as session:
        enter = types.SimpleNamespace(
            callback_site=FakeApiCallbackSite.API_ENTER, function_name="cudaMalloc"
        )
        exit_site = types.SimpleNamespace(
            callback_site=FakeApiCallbackSite.API_EXIT, function_name="cudaMalloc"
        )
        fake_cupti._cb(None, None, 0, enter)
        fake_cupti._cb(None, None, 0, enter)
        fake_cupti._cb(None, None, 0, exit_site)  # EXIT is not counted
    assert session.counts() == {"cudaMalloc": 2}
    assert fake_cupti.subscribed == []  # stop unsubscribed


def test_nvtx_tracer_open_and_callbacks_use_cupti(fake_cupti: FakeCupti) -> None:
    """With CUPTI present, the tracer builds CUPTI-backed collectors and sessions."""
    tracer = nv.NvtxTracer()
    assert isinstance(tracer.open(Activity.KERNEL), nv.CuptiCollector)
    assert isinstance(tracer.callbacks(), nv.CuptiCallbackSession)


def test_collector_stop_is_safe_when_not_the_active_one(fake_cupti: FakeCupti) -> None:
    """Stopping a collector that is not on top of the active stack leaves the stack alone."""
    collector = nv.CuptiCollector(Activity.KERNEL)
    collector.stop()  # never entered, so not active -> just flushes, pops nothing
    assert nv._active == []


def test_callback_session_stop_is_idempotent(fake_cupti: FakeCupti) -> None:
    """Stopping a callback session twice is safe: the second stop is a no-op."""
    session = nv.CuptiCallbackSession(("runtime",))
    session.__enter__()
    session.stop()
    session.stop()  # subscriber already cleared -> no-op
    assert fake_cupti.subscribed == []
