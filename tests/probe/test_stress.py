from mainboard.probe.stress import Precision, Stress, StressReport, rows


class FakeKernels:
    """Deterministic device operations: every GEMM takes one millisecond, FP8 is unsupported."""

    def __init__(self) -> None:
        self.synchronized = 0

    def describe(self) -> tuple[str, tuple[int, int], int, int]:
        return "Fake Card", (8, 9), 128, 2_565_000

    def gemm(self, precision: Precision, n: int):
        if precision == Precision.FP8:
            raise RuntimeError("fp8 needs sm_89")
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


def test_rows_and_round_trip() -> None:
    report = Stress(FakeKernels(), n=256, megabytes=8, warmups=0, repetitions=1).measure()
    again = StressReport.model_validate_json(report.model_dump_json())
    assert again == report
    listed = rows(report)
    assert [row["measure"] for row in listed][:2] == ["FP64", "FP32"]
    assert [row["unit"] for row in listed][-3:] == ["GB/s"] * 3
