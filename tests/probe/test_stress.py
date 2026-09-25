import runpy
import sys
import time

import pytest

from mainboard.probe import GPU, NvidiaGPU
from mainboard.probe.stress import Precision, Stress, StressReport, TorchKernels

from .support import FakeNvidiaApis, FakeTorch


class FakeKernels:
    """Deterministic device operations: every GEMM takes one millisecond, FP8 is unsupported."""

    def __init__(self) -> None:
        self.synchronized = 0

    def describe(self) -> tuple[str, tuple[int, int], int, int]:
        return "Fake Card", (8, 9), 128, 2_565_000

    def gemm(self, precision: Precision, n: int):
        if precision == Precision.FP8:
            raise RuntimeError("fp8 needs sm_89")
        if precision == Precision.INT8:

            def refused() -> None:
                raise RuntimeError("int8 refused at call time")

            return refused
        return lambda: None

    def copy(self, path: str, megabytes: int):
        return lambda: None

    def synchronize(self) -> None:
        self.synchronized += 1


def test_report_carries_every_precision_and_link(monkeypatch) -> None:
    ticks = iter(range(10_000))
    monkeypatch.setattr("mainboard.probe.stress.perf_counter", lambda: next(ticks) * 1e-3)
    kernels = FakeKernels()
    report = Stress(kernels, n=1024, megabytes=64, warmups=1, repetitions=3).measure()
    assert [rate.precision for rate in report.rates] == list(Precision)
    assert report.rate(Precision.FP64).n == 512
    assert not report.rate(Precision.FP8).supported
    assert "sm_89" in report.rate(Precision.FP8).note
    assert not report.rate(Precision.INT8).supported
    assert "call time" in report.rate(Precision.INT8).note
    fp32 = report.rate(Precision.FP32)
    assert fp32.seconds > 0 and fp32.tflops == 2 * 1024**3 / fp32.seconds / 1e12
    assert [link.path for link in report.links] == [
        "device_to_device",
        "host_to_device",
        "device_to_host",
    ]
    assert report.links[0].gb_s == 2 * report.links[1].gb_s
    assert report.datasheet_fp32_tflops == 128 * 128 * 2 * 2_565_000 * 1e3 / 1e12
    assert kernels.synchronized > 0


def test_the_report_round_trips_through_its_json() -> None:
    report = Stress(FakeKernels(), n=256, megabytes=8, warmups=0, repetitions=1).measure()
    assert StressReport.model_validate_json(report.model_dump_json()) == report


@pytest.mark.parametrize(
    ("index", "clock_khz"), [(0, 2_520_000), (1, 0)], ids=["probed", "unprobed"]
)
def test_torch_kernels_describe_the_card_with_the_probed_peak_clock(
    index: int,
    clock_khz: int,
    fake_torch: FakeTorch,
    nvidia_host: FakeNvidiaApis,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The datasheet clock comes from the host probe, and a device it did not find has none."""
    monkeypatch.setattr(GPU, "all", staticmethod(lambda: (NvidiaGPU(index=0),)))
    kernels = TorchKernels(index)
    assert fake_torch.cuda.current == f"cuda:{index}"
    assert kernels.describe() == ("NVIDIA GeForce RTX 4090", (8, 9), 128, clock_khz)


def test_torch_kernels_reach_each_precisions_own_library_routine(fake_torch: FakeTorch) -> None:
    """TF32 is FP32 with the tensor-core switch on, FP8 is scaled into BF16, INT8 is integer.

    Every copy moves between the right ends, and the host side of a transfer is asynchronous.
    """
    kernels = TorchKernels()
    for precision in Precision:
        kernels.gemm(precision, 4)()
    for path in ("device_to_device", "host_to_device", "device_to_host"):
        kernels.copy(path, 1)()
    kernels.synchronize()
    assert fake_torch.calls == [
        ("matmul", "float64", False),
        ("matmul", "float32", False),
        ("matmul", "float32", True),
        ("matmul", "float16", True),
        ("matmul", "bfloat16", True),
        ("_scaled_mm", "float8_e4m3fn", "bfloat16"),
        ("_int_mm", "int8"),
        ("copy", "cuda:0", "cuda:0", False),
        ("copy", "cpu", "cuda:0", True),
        ("copy", "cuda:0", "cpu", True),
    ]
    assert fake_torch.cuda.synchronized == 1


# Running the imported module again as `__main__` is the point, and the entry reads `sys.argv`.
@pytest.mark.filterwarnings("ignore:'mainboard.probe.stress' found in sys.modules:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:Cyclopts application invoked without tokens:UserWarning")
def test_the_module_entry_prints_the_report_a_dispatched_probe_reads_back(
    fake_torch: FakeTorch,
    nvidia_host: FakeNvidiaApis,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`python -m mainboard.probe.stress` measures the named device and prints one JSON line."""
    ticks = iter(range(10_000))
    monkeypatch.setattr(time, "perf_counter", lambda: next(ticks) * 1e-3)
    monkeypatch.setattr(GPU, "all", staticmethod(lambda: (NvidiaGPU(index=0),)))
    monkeypatch.setattr(sys, "argv", ["stress", "--n", "64", "--repetitions", "1"])
    with pytest.raises(SystemExit, match="^0$"):
        runpy.run_module("mainboard.probe.stress", run_name="__main__", alter_sys=True)
    report = StressReport.model_validate_json(capsys.readouterr().out)
    assert (report.device, report.capability, report.clock_khz) == (
        "NVIDIA GeForce RTX 4090",
        "8.9",
        2_520_000,
    )
    assert all(rate.supported for rate in report.rates)
    assert report.rates[0].n == 32
