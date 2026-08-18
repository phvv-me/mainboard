import pytest

from mainboard.probe import gating


class _Gpu:
    def __init__(self, busy: bool) -> None:
        self.busy = busy


@pytest.fixture
def fleet(monkeypatch: pytest.MonkeyPatch) -> list[_Gpu]:
    gpus: list[_Gpu] = []
    monkeypatch.setattr(gating.Machine, "gpus", property(lambda self: tuple(gpus)))
    monkeypatch.setattr(
        gating, "device_busy", lambda gpu, *, util_threshold, memory_threshold_pct: bool(gpu and gpu.busy)
    )
    return gpus


def test_gpu_busy_reads_the_first_visible_device(fleet: list[_Gpu]) -> None:
    assert gating.gpu_busy() is False
    fleet.append(_Gpu(busy=True))
    assert gating.gpu_busy() is True
    fleet[0] = _Gpu(busy=False)
    assert gating.gpu_busy() is False


def test_wait_for_idle_returns_once_idle_holds(fleet: list[_Gpu]) -> None:
    fleet.append(_Gpu(busy=False))
    assert gating.wait_for_idle(timeout=1.0, idle_duration=0.0, poll_interval=0.01) is True


def test_wait_for_idle_times_out_while_busy(fleet: list[_Gpu]) -> None:
    fleet.append(_Gpu(busy=True))
    assert gating.wait_for_idle(timeout=0.05, poll_interval=0.01) is False


def test_wait_for_idle_resets_when_business_interrupts(fleet: list[_Gpu]) -> None:
    fleet.append(_Gpu(busy=False))
    flips = iter([False, True, False, False, False, False])
    gating_module = gating

    def flipping(gpu, *, util_threshold, memory_threshold_pct):
        return next(flips, False)

    gating_module.device_busy = flipping
    assert gating.wait_for_idle(timeout=1.0, idle_duration=0.02, poll_interval=0.01) is True
