"""What the CUPTI buffer callback keeps and what retrieval hands back, with no GPU: the callback
only reads attributes of the activity objects CUPTI passes it."""

from types import SimpleNamespace

import pytest

from mainboard.profile.providers.nvidia import tracer
from mainboard.profile.trace import KernelTrace, MemcpyTrace

SPAN = {"start": 1_000, "end": 1_900, "correlation_id": 7, "device_id": 0, "context_id": 1}
KERNEL = SimpleNamespace(
    kind=tracer._CONCURRENT_KERNEL,
    name="gemm",
    grid_x=128,
    grid_y=1,
    grid_z=1,
    block_x=256,
    block_y=2,
    block_z=1,
    static_shared_memory=16,
    dynamic_shared_memory=4096,
    registers_per_thread=32,
    stream_id=3,
    **SPAN,
)
MEMCPY = SimpleNamespace(kind=tracer._MEMCPY, copy_kind=1, bytes=4096, stream_id=3, **SPAN)
RUNTIME = SimpleNamespace(kind=99, name=None, cbid=None, **SPAN)


@pytest.fixture
def collector(monkeypatch) -> tracer.CuptiCollector:
    """A running collector standing in for the active one, its context (device 0, context 1)."""
    found = tracer.CuptiCollector()
    found.running, found.scope = True, (0, 0, 1)
    monkeypatch.setattr(tracer, "_active", [found])
    monkeypatch.setitem(tracer._label, 99, "runtime")
    tracer._on_buffer_completed([KERNEL, MEMCPY, RUNTIME])
    return found


def test_each_record_is_validated_once_and_retrieved_as_kept(collector) -> None:
    [kernel] = collector.kernels(since=0)
    [memcpy] = collector.memcpys(since=0)

    assert kernel is collector.records[0] and isinstance(kernel, KernelTrace)
    assert (kernel.grid, kernel.block, kernel.threads_per_block) == ("128x1x1", "256x2x1", 512)
    assert (kernel.name, kernel.duration_ns, kernel.stream_id) == ("gemm", 900, 3)
    assert memcpy is collector.records[1] and isinstance(memcpy, MemcpyTrace)
    assert (memcpy.kind, memcpy.bytes_moved) == ("HtoD", 4096)
    assert [record.kind for record in collector.activities()] == ["runtime"]


def test_a_window_holding_another_context_is_refused(collector) -> None:
    collector.scope = (0, 0, 2)

    with pytest.raises(RuntimeError, match="foreign or unknown CUDA context"):
        collector.kernels(since=0)
