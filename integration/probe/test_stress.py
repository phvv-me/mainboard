from types import SimpleNamespace

import pytest

from mainboard.probe.stress import Stress

from .constants import OPERATION_SECONDS, UNSUPPORTED
from .fakes import CountedKernels

_WARMUPS, _REPETITIONS = 2, 3


@pytest.fixture
def kernels(monkeypatch: pytest.MonkeyPatch) -> CountedKernels:
    """A counted device, built where the framework is imported."""
    torch = SimpleNamespace(
        device=lambda kind, index: (kind, index), cuda=SimpleNamespace(set_device=lambda _: None)
    )
    monkeypatch.setattr("mainboard.probe.stress.import_module", lambda name: torch)
    return CountedKernels()


def test_every_precision_and_copy_path_is_timed_after_its_warmups(kernels: CountedKernels) -> None:
    report = Stress(kernels, n=64, warmups=_WARMUPS, repetitions=_REPETITIONS).measure()

    timed = {rate.precision: rate for rate in report.rates if rate.supported}
    paths = [link.path for link in report.links]
    assert len(timed) == len(report.rates) - 1 and UNSUPPORTED not in timed
    assert all(OPERATION_SECONDS <= rate.seconds < 1 for rate in timed.values())
    assert all(rate.tflops > 0 for rate in timed.values())
    assert len(paths) == 3
    assert all(link.gb_s > 0 for link in report.links)
    assert {kernels.calls[name] for name in (*timed, *paths)} == {_WARMUPS + _REPETITIONS}
    assert kernels.calls["synchronize"] >= len(timed) + len(paths)


def test_a_precision_whose_call_raises_is_noted_by_the_first_line_of_why(
    kernels: CountedKernels,
) -> None:
    report = Stress(kernels, n=64, warmups=_WARMUPS, repetitions=_REPETITIONS).measure()

    [refused] = [rate for rate in report.rates if not rate.supported]
    assert (refused.precision, refused.seconds, refused.tflops) == (UNSUPPORTED, 0.0, 0.0)
    assert refused.note == "no such precision"
    assert kernels.calls[UNSUPPORTED] == 1
