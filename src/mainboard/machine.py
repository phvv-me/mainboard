from __future__ import annotations

import platform
from functools import cached_property

from patos import Singleton

from .cpu import CPU
from .gpu import GPU
from .host import Host
from .models.board import Board
from .models.environment import Environment
from .models.machine_snapshot import CpuSnapshot, MachineSnapshot
from .models.system_compilers import SystemCompilers
from .models.toolchain import Toolchain
from .npu import NPU
from .providers import NvidiaGPU
from .unit import Unit


class Machine(Singleton):
    """Singleton facade for the host and hardware units."""

    @cached_property
    def host(self) -> Host:
        """Detected host CPU, memory, and disk."""
        return Host()

    @cached_property
    def cpu(self) -> CPU:
        """Detected host CPU."""
        return CPU(
            name_value=self.host.cpu,
            architecture_value=self.host.arch,
            logical_cores=self.host.logical_cpus,
            physical_cores=self.host.physical_cpus,
            current_clock_mhz=self.host.cpu_freq_mhz,
            vendor=self.host.cpu_vendor,
        )

    @cached_property
    def gpus(self) -> tuple[GPU, ...]:
        """Detected GPUs across supported providers."""
        return GPU.all()

    @cached_property
    def npus(self) -> tuple[NPU, ...]:
        """Detected neural processing units."""
        return NPU.all()

    @cached_property
    def environment(self) -> Environment:
        """The host's execution context: user, group(s), and job scheduler on PATH."""
        return Environment.probe()

    @cached_property
    def board(self) -> Board:
        """The host's motherboard and firmware identity."""
        return Board.probe()

    @cached_property
    def toolchain(self) -> Toolchain:
        """C/C++/CUDA compilers and build systems found on the host PATH."""
        return Toolchain.probe()

    @cached_property
    def units(self) -> tuple[Unit, ...]:
        """All detected schedulable units."""
        return (self.cpu, *self.gpus, *self.npus)

    @cached_property
    def compilers(self) -> SystemCompilers:
        """Detected host compilers with CMake build configuration."""
        cuda_gpus = [gpu for gpu in self.gpus if isinstance(gpu, NvidiaGPU)]
        if not cuda_gpus:
            raise RuntimeError("No CUDA devices detected; CUDA compiler settings are unavailable.")
        cc = max(gpu.cuda_architecture for gpu in cuda_gpus)
        return SystemCompilers(
            arch=self.host.arch,
            cpu=self.host.cpu,
            cuda_arch=f"{cc.major}{cc.minor}",
        )

    @cached_property
    def nvcc_path(self) -> str:
        """Absolute path to the `nvcc` binary."""
        return str(self.compilers.nvcc.path)

    @cached_property
    def cuda_architecture(self) -> str:
        """CUDA compute capability without a dot, e.g. `89`."""
        return self.compilers.cuda_arch

    def snapshot(self) -> MachineSnapshot:
        """Probe the host's compute resources into one serializable model.

        Provider detection is best-effort: a host with no accelerator yields
        empty `gpus` and `npus` rather than raising.
        """
        cpu, memory = self.cpu, self.host.memory
        return MachineSnapshot(
            hostname=platform.node(),
            cpu=CpuSnapshot(
                name=cpu.name,
                architecture=cpu.architecture,
                vendor=cpu.vendor,
                logical_cores=cpu.logical_cores,
                physical_cores=cpu.physical_cores,
                total_memory_bytes=memory.total_bytes,
                current_clock_mhz=cpu.current_clock_mhz,
            ),
            memory=memory,
            environment=self.environment,
            board=self.board,
            toolchain=self.toolchain,
            gpus=tuple(gpu.snapshot() for gpu in self.gpus),
            npus=tuple(npu.snapshot() for npu in self.npus),
        )

    def model_dump_json(self, *, indent: int | None = None) -> str:
        """Probe the host and serialize the snapshot to JSON.

        indent: pretty-print indent passed through to pydantic when set.
        """
        return self.snapshot().model_dump_json(indent=indent)
