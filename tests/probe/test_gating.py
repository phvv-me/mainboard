from collections.abc import Iterator

import pytest

from mainboard.probe import gating


class FakeGpu:
    """A device that is busy or not, the only thing the gate reads off one."""

    def __init__(self, busy: bool) -> None:
        self.busy = busy


class FakeClock:
    """A monotonic clock that only moves when the code under test sleeps on it.

    The wait loop is pure bookkeeping over elapsed time, so driving it from here keeps the test
    exact and instant instead of spending real seconds and hoping the scheduler cooperates.
    """

    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def fleet(monkeypatch: pytest.MonkeyPatch) -> list[FakeGpu]:
    """A mutable stand-in for `Machine().gpus` whose devices the busy check reads."""
    gpus: list[FakeGpu] = []
    monkeypatch.setattr(gating, "time", FakeClock())
    monkeypatch.setattr(gating.Machine, "gpus", property(lambda self: tuple(gpus)))
    monkeypatch.setattr(
        gating,
        "device_busy",
        lambda gpu, *, util_threshold, memory_threshold_pct: bool(gpu and gpu.busy),
    )
    return gpus


def test_gpu_busy_reads_the_first_visible_device_and_calls_a_bare_host_idle(
    fleet: list[FakeGpu],
) -> None:
    """A host with no GPU at all has nothing to wait on, so it reads idle rather than raising."""
    assert gating.gpu_busy() is False
    fleet.append(FakeGpu(busy=True))
    assert gating.gpu_busy() is True
    fleet[0] = FakeGpu(busy=False)
    assert gating.gpu_busy() is False


def test_wait_for_idle_holds_out_for_a_continuous_idle_window(
    fleet: list[FakeGpu], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only an unbroken idle stretch returns.

    A single idle sample is not enough, since a job between kernels looks idle for an
    instant, so any busy reading restarts the window.
    """
    fleet.append(FakeGpu(busy=False))
    readings: Iterator[bool] = iter([False, True, False])
    monkeypatch.setattr(
        gating,
        "device_busy",
        lambda gpu, *, util_threshold, memory_threshold_pct: next(readings, False),
    )
    assert gating.wait_for_idle(timeout=30.0, idle_duration=1.0, poll_interval=0.5) is True


def test_wait_for_idle_gives_up_at_the_deadline_while_the_device_stays_busy(
    fleet: list[FakeGpu],
) -> None:
    """The wait is bounded, so a device that never frees up returns a refusal, not a hang."""
    fleet.append(FakeGpu(busy=True))
    assert gating.wait_for_idle(timeout=2.0, poll_interval=0.5) is False
