import time
from collections.abc import Sequence
from types import TracebackType
from typing import Protocol


class MemoryReading(Protocol):
    """The memory figure the meter samples."""

    @property
    def used_gb(self) -> float: ...


class MemorySource(Protocol):
    """A unit or host that exposes a `memory` reading."""

    @property
    def memory(self) -> MemoryReading: ...


class MeteredMachine(Protocol):
    """The slice of a machine the meter samples: its host and GPUs."""

    @property
    def gpus(self) -> Sequence[MemorySource]: ...

    @property
    def host(self) -> MemorySource: ...


class Meter:
    """Times a region and tracks peak host and GPU memory, in gibibytes, across samples.

    Memory is sampled at enter, at every explicit `sample()`, and at exit; no background
    thread runs. mainboard.probe is not a dependency of profiling, so the caller passes its
    own `Machine()` or a stand-in.
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

    @property
    def host_delta_gb(self) -> float:
        """Host memory growth from the first to the last sample."""
        return self.host_used_gb[-1] - self.host_used_gb[0] if self.host_used_gb else 0.0

    @property
    def peak_gpu_gb(self) -> float:
        """Highest total GPU memory in use across samples."""
        return max(self.gpu_used_gb, default=0.0)

    @property
    def peak_host_gb(self) -> float:
        return max(self.host_used_gb, default=0.0)

    def sample(self) -> None:
        """Capture one host and one summed GPU memory reading from the live machine."""
        self.host_used_gb.append(self.machine.host.memory.used_gb)
        self.gpu_used_gb.append(sum(gpu.memory.used_gb for gpu in self.machine.gpus))
