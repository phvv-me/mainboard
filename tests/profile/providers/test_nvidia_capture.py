"""Capture barriers and loss accounting, without CUDA initialization."""

import types
from typing import TYPE_CHECKING

import pytest

from mainboard.profile import Activity
from mainboard.profile.providers import nvidia_tracer as nv

from .test_nvidia_tracer import fake_cupti as fake_cupti

if TYPE_CHECKING:
    from .test_nvidia_tracer import FakeCupti


def test_device_sync_requires_the_runtime_binding(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing barrier cannot be reported as a complete capture."""
    monkeypatch.setattr(nv, "_runtime_loaded", True)
    monkeypatch.setattr(nv, "cuda_runtime", None)
    with pytest.raises(RuntimeError, match="runtime binding is required"):
        nv._sync()


@pytest.mark.parametrize("status", [0, 1, 999])
def test_device_sync_checks_the_return_status(
    monkeypatch: pytest.MonkeyPatch, status: int
) -> None:
    """Only cudaSuccess completes the barrier; launch errors remain failures."""
    monkeypatch.setattr(nv, "_runtime_loaded", True)
    synced: list[int] = []

    def synchronize() -> tuple[int]:
        synced.append(1)
        return (status,)

    monkeypatch.setattr(
        nv, "cuda_runtime", types.SimpleNamespace(cudaDeviceSynchronize=synchronize)
    )
    if status:
        with pytest.raises(RuntimeError, match=f"CUDA status {status}"):
            nv._sync()
    else:
        nv._sync()
    assert synced == [1]


def test_native_loss_is_added_once_even_without_delivered_records(fake_cupti: FakeCupti) -> None:
    """Reset-on-read native loss is distinct from overwritten Python records."""
    fake_cupti.native_dropped = 17  # stale activity before this capture is excluded
    with nv.CuptiCollector(Activity.KERNEL, max_records=1) as collector:
        assert collector.dropped() == 0
        fake_cupti.native_dropped = 3
        collector.flush()  # no completion callback: all records may have been lost
        assert collector.native_dropped_records == 3
        assert collector.dropped_records == 0
        collector.flush()
        assert collector.dropped() == 3
        record = nv.RawMemcpy(copy_kind=1, start_ns=0, end_ns=1, bytes_moved=1, correlation_id=0)
        collector.append(record)
        collector.append(record)
        fake_cupti.native_dropped = 2
    assert collector.native_dropped_records == 5
    assert collector.dropped_records == 1
    assert collector.dropped() == 6
    assert fake_cupti.drop_queries == [(0, 0)] * 4


def test_reset_discards_both_loss_counts_at_the_same_boundary(fake_cupti: FakeCupti) -> None:
    """Reset drains pending native loss before clearing the measurement window."""
    with nv.CuptiCollector(Activity.KERNEL, max_records=1) as collector:
        record = nv.RawMemcpy(copy_kind=1, start_ns=0, end_ns=1, bytes_moved=1, correlation_id=0)
        collector.append(record)
        collector.append(record)
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
