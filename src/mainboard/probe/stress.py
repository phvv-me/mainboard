"""Achieved throughput of one device, measured rather than read from a datasheet.

A datasheet peak is multiprocessors times cores times two operations per boost clock, which
no kernel reaches: Luo et al. (arXiv 2402.13499) measure 63 to 95 percent of it at the
instruction level depending on the instruction family, and a library GEMM lands below the
instruction ceiling again. This probe measures the library level, one square GEMM per
precision through the framework's own matrix multiply, and the three copies a call pays:
device to device, pinned host to device, device to host. Every number is a median of timed
repetitions after warmups, computed as operations over wall time so a card's real boost clock
is what is measured, never assumed.

The framework is imported when a measurement runs, so the probe is importable on a machine
that carries no accelerator stack and refuses by naming what is missing.
"""

from __future__ import annotations

from enum import StrEnum, auto
from statistics import median
from time import perf_counter
from typing import TYPE_CHECKING, Protocol

from cyclopts import App
from patos import FrozenModel, FrozenOpenModel

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

_SCHEMA_VERSION = 1
# FP32 lanes per multiprocessor by compute capability major version; the datasheet peak the
# achieved rate is compared against. Volta and Turing (7) carry 64, Ampere data-center (8.0)
# too, and every later part 128.
_LANES_PER_SM = {7: 64, 8: 128, 9: 128, 10: 128, 12: 128}
_AMPERE_DATACENTER = (8, 0)


class Precision(StrEnum):
    """The precisions a GEMM is timed at, tensor-core ones included."""

    FP64 = auto()
    FP32 = auto()
    TF32 = auto()
    FP16 = auto()
    BF16 = auto()
    FP8 = auto()
    INT8 = auto()


class Rate(FrozenModel):
    """One precision's achieved rate.

    precision: the operand precision.
    n: the square GEMM's side; operations are two n cubed.
    seconds: the median wall time of one GEMM, zero when unsupported.
    tflops: achieved tera-operations per second, zero when unsupported.
    supported: whether the device and framework run this precision at all.
    note: why an unsupported precision was skipped.
    """

    precision: Precision
    n: int
    seconds: float = 0.0
    tflops: float = 0.0
    supported: bool = True
    note: str = ""


class Link(FrozenModel):
    """One copy path's achieved bandwidth.

    path: `device_to_device`, `host_to_device` or `device_to_host`.
    megabytes: the buffer copied.
    gb_s: bytes moved over wall time; a device copy counts its read and its write.
    """

    path: str
    megabytes: int
    gb_s: float


class StressReport(FrozenOpenModel):
    """One device's measured limits, the JSON another machine reads back.

    schema_version: format revision, bumped when a field's meaning changes.
    device: the card's name.
    capability: compute capability as `major.minor`.
    sm_count: multiprocessors.
    clock_khz: the maximum SM clock the device reports, which the datasheet peak uses.
    datasheet_fp32_tflops: multiprocessors times lanes times two per clock.
    rates: one entry per precision, in the order measured.
    links: the three copy paths.
    seconds: wall time of the whole probe.
    """

    schema_version: int = _SCHEMA_VERSION
    device: str = ""
    capability: str = ""
    sm_count: int = 0
    clock_khz: int = 0
    datasheet_fp32_tflops: float = 0.0
    rates: tuple[Rate, ...] = ()
    links: tuple[Link, ...] = ()
    seconds: float = 0.0

    def rate(self, precision: Precision) -> Rate:
        """The entry for one precision."""
        return next(rate for rate in self.rates if rate.precision == precision)


class Kernels(Protocol):
    """The device operations a stress measurement times, one framework behind them."""

    def describe(self) -> tuple[str, tuple[int, int], int, int]:
        """Name, compute capability, multiprocessor count and clock in kHz."""
        ...

    def gemm(self, precision: Precision, n: int) -> Callable[[], None]:
        """A closure running one n by n GEMM at `precision`, or raise when unsupported."""
        ...

    def copy(self, path: str, megabytes: int) -> Callable[[], None]:
        """A closure moving `megabytes` along `path`."""
        ...

    def synchronize(self) -> None:
        """Wait for every queued device operation."""
        ...


class TorchKernels:
    """The stress operations through PyTorch, imported when first used."""

    def __init__(self, device_index: int = 0) -> None:
        import torch

        self.torch = torch
        self.device = torch.device("cuda", device_index)
        self.index = device_index
        torch.cuda.set_device(self.device)

    def describe(self) -> tuple[str, tuple[int, int], int, int]:
        from .machine import Machine

        properties = self.torch.cuda.get_device_properties(self.index)
        capability = (int(properties.major), int(properties.minor))
        gpus = Machine().gpus
        clock = gpus[self.index].peak_clock_khz if self.index < len(gpus) else 0
        return properties.name, capability, int(properties.multi_processor_count), clock

    def gemm(self, precision: Precision, n: int) -> Callable[[], None]:
        torch = self.torch
        if precision in {Precision.FP32, Precision.TF32}:
            torch.backends.cuda.matmul.allow_tf32 = precision == Precision.TF32
            left = torch.randn(n, n, device=self.device)
            right = torch.randn(n, n, device=self.device)
            return lambda: torch.matmul(left, right)
        if precision == Precision.INT8:
            shape = (n, n)
            left = torch.randint(-128, 127, shape, device=self.device, dtype=torch.int8)
            right = torch.randint(-128, 127, shape, device=self.device, dtype=torch.int8)
            return lambda: torch._int_mm(left, right)
        if precision == Precision.FP8:
            source = torch.randn(n, n, device=self.device)
            left = source.to(torch.float8_e4m3fn)
            right = source.t().contiguous().t().to(torch.float8_e4m3fn)
            scale = torch.ones((), device=self.device)
            return lambda: torch._scaled_mm(
                left, right, scale_a=scale, scale_b=scale, out_dtype=torch.bfloat16
            )
        dtype = {
            Precision.FP64: torch.float64,
            Precision.FP16: torch.float16,
            Precision.BF16: torch.bfloat16,
        }[precision]
        left = torch.randn(n, n, device=self.device, dtype=dtype)
        right = torch.randn(n, n, device=self.device, dtype=dtype)
        return lambda: torch.matmul(left, right)

    def copy(self, path: str, megabytes: int) -> Callable[[], None]:
        torch = self.torch
        count = megabytes << 20
        device = torch.empty(count, dtype=torch.uint8, device=self.device)
        if path == "device_to_device":
            source = torch.empty_like(device)
            return lambda: device.copy_(source)
        host = torch.empty(count, dtype=torch.uint8, pin_memory=True)
        if path == "host_to_device":
            return lambda: device.copy_(host, non_blocking=True)
        return lambda: host.copy_(device, non_blocking=True)

    def synchronize(self) -> None:
        self.torch.cuda.synchronize(self.device)


class Stress:
    """Measure one device's achieved rates and link bandwidths.

    kernels: the device operations, PyTorch's unless a test injects its own.
    n: the square GEMM side for every precision but FP64, which runs at half the side
        because a consumer card's FP64 rate would make the full size take seconds.
    megabytes: the copy buffer.
    warmups: untimed calls before the timed ones.
    repetitions: timed calls whose median is kept.
    """

    def __init__(
        self,
        kernels: Kernels | None = None,
        *,
        n: int = 8192,
        megabytes: int = 256,
        warmups: int = 2,
        repetitions: int = 5,
    ) -> None:
        self.kernels = kernels if kernels is not None else TorchKernels()
        self.n = n
        self.megabytes = megabytes
        self.warmups = warmups
        self.repetitions = repetitions

    def measure(self) -> StressReport:
        """Time every precision and copy path, answering the report."""
        start = perf_counter()
        device, capability, sm_count, clock_khz = self.kernels.describe()
        lanes = 64 if capability == _AMPERE_DATACENTER else _LANES_PER_SM.get(capability[0], 128)
        rates = tuple(self._rate(precision) for precision in Precision)
        links = tuple(
            self._link(path) for path in ("device_to_device", "host_to_device", "device_to_host")
        )
        return StressReport(
            device=device,
            capability=f"{capability[0]}.{capability[1]}",
            sm_count=sm_count,
            clock_khz=clock_khz,
            datasheet_fp32_tflops=sm_count * lanes * 2 * clock_khz * 1e3 / 1e12,
            rates=rates,
            links=links,
            seconds=perf_counter() - start,
        )

    def _rate(self, precision: Precision) -> Rate:
        n = self.n // 2 if precision == Precision.FP64 else self.n
        try:
            seconds = self._timed(self.kernels.gemm(precision, n))
        except (RuntimeError, NotImplementedError, AttributeError, KeyError) as error:
            return Rate(precision=precision, n=n, supported=False, note=str(error).splitlines()[0])
        return Rate(precision=precision, n=n, seconds=seconds, tflops=2 * n**3 / seconds / 1e12)

    def _link(self, path: str) -> Link:
        seconds = self._timed(self.kernels.copy(path, self.megabytes))
        moved = (self.megabytes << 20) * (2 if path == "device_to_device" else 1)
        return Link(path=path, megabytes=self.megabytes, gb_s=moved / seconds / 1e9)

    def _timed(self, call: Callable[[], None]) -> float:
        """The median wall time of `call` over the timed repetitions, device synchronized."""
        for _ in range(self.warmups):
            call()
        self.kernels.synchronize()
        timings = []
        for _ in range(self.repetitions):
            self.kernels.synchronize()
            begin = perf_counter()
            call()
            self.kernels.synchronize()
            timings.append(perf_counter() - begin)
        return median(timings)


app = App(name="stress", help="Measure this device's achieved rates and print the report JSON.")


@app.default
def main(*, device: int = 0, n: int = 8192, repetitions: int = 5) -> None:
    """Measure the device and print the report as JSON, for a dispatched probe.

    device: the CUDA device index.
    n: the square GEMM side.
    repetitions: timed calls per measurement.
    """
    report = Stress(TorchKernels(device), n=n, repetitions=repetitions).measure()
    print(report.model_dump_json())


def rows(report: StressReport) -> Sequence[dict[str, str | float | int]]:
    """The report as one row per precision and link, for tables."""
    listed: list[dict[str, str | float | int]] = [
        {
            "measure": rate.precision.upper(),
            "n": rate.n,
            "value": round(rate.tflops, 1) if rate.supported else 0.0,
            "unit": "TOPS" if rate.precision == Precision.INT8 else "TFLOPS",
            "note": rate.note if not rate.supported else "",
        }
        for rate in report.rates
    ]
    listed.extend(
        {
            "measure": link.path,
            "n": link.megabytes,
            "value": round(link.gb_s, 1),
            "unit": "GB/s",
            "note": "",
        }
        for link in report.links
    )
    return listed


if __name__ == "__main__":
    app()
