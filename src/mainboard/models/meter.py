import time
from collections.abc import Sequence
from types import TracebackType
from typing import Protocol

from .memory import Memory


class MemorySource(Protocol):
    """A unit or host that exposes a `memory` reading."""

    @property
    def memory(self) -> Memory: ...


class MeteredMachine(Protocol):
    """The slice of a machine the meter samples: its host and GPUs."""

    @property
    def host(self) -> MemorySource: ...

    @property
    def gpus(self) -> Sequence[MemorySource]: ...


class Meter:
    """Times a region and tracks peak host and GPU memory across samples.

    Memory is sampled from the live `Machine` snapshot at enter, at every
    explicit `sample()`, and at exit; peaks are the maximum used bytes over
    all samples. No background thread is used: callers drive sampling.
    """

    def __init__(self, machine: MeteredMachine) -> None:
        self.machine = machine
        self.host_used_gb: list[float] = []
        self.gpu_used_gb: list[float] = []
        self.start_ns = 0
        self.elapsed_s = 0.0

    def __enter__(self) -> Meter:
        self.start_ns = time.perf_counter_ns()
        self.sample()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.sample()
        self.elapsed_s = (time.perf_counter_ns() - self.start_ns) / 1e9

    def sample(self) -> None:
        """Capture one host and GPU memory reading from the live machine."""
        self.host_used_gb.append(self.machine.host.memory.used_gb)
        self.gpu_used_gb.append(sum(gpu.memory.used_gb for gpu in self.machine.gpus))

    @property
    def peak_host_gb(self) -> float:
        """Highest host memory in use across samples, in gibibytes."""
        return max(self.host_used_gb, default=0.0)

    @property
    def peak_gpu_gb(self) -> float:
        """Highest total GPU memory in use across samples, in gibibytes."""
        return max(self.gpu_used_gb, default=0.0)

    @property
    def host_delta_gb(self) -> float:
        """Host memory growth from the first to the last sample, in gibibytes."""
        if not self.host_used_gb:
            return 0.0
        return self.host_used_gb[-1] - self.host_used_gb[0]


def meter() -> Meter:
    """Open a runtime-metrics meter bound to the current machine.

    Use as `with meter() as m: ...`, then read `m.elapsed_s`,
    `m.peak_host_gb`, `m.peak_gpu_gb`, and `m.host_delta_gb`.
    """
    from ..machine import Machine

    return Meter(Machine())
