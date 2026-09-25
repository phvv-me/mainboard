from functools import cached_property
from typing import TYPE_CHECKING

from patos import Singleton

from .facts.compilers import Compilers
from .facts.environment import Environment
from .host import Host
from .providers.nvidia import NvidiaGPU
from .units.cpu import CPU
from .units.gpu import GPU
from .units.npu import NPU

if TYPE_CHECKING:
    from .units.unit import Unit


class Machine(Singleton):
    """Singleton facade for the host and hardware units.

    Every subsystem is best-effort: a host with no accelerator, no scheduler, or no cgroup cap
    answers with an empty or zeroed value rather than raising. `compilers` is the one exception.
    """

    @cached_property
    def compilers(self) -> Compilers:
        """Host C++ and CUDA compilers targeting the newest CUDA device on this machine.

        Raises `RuntimeError` without a CUDA device, the compute capability the build targets
        being read off the detected GPUs.
        """
        capabilities = [gpu.cuda_architecture for gpu in self.gpus if isinstance(gpu, NvidiaGPU)]
        if not capabilities:
            raise RuntimeError("No CUDA device detected, so compiler settings are unavailable.")
        target = max(capabilities)
        return Compilers(
            arch=self.host.arch, cpu=self.host.cpu, cuda_arch=f"{target.major}{target.minor}"
        )

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
    def environment(self) -> Environment:
        """The host's execution context, the job scheduler on PATH."""
        return Environment.probe()

    @cached_property
    def gpus(self) -> tuple[GPU, ...]:
        """Detected GPUs across supported providers."""
        return GPU.all()

    @cached_property
    def host(self) -> Host:
        """Detected host CPU, memory, and disk."""
        return Host()

    @cached_property
    def npus(self) -> tuple[NPU, ...]:
        """Detected neural processing units."""
        return NPU.all()

    @cached_property
    def units(self) -> tuple[Unit, ...]:
        """All detected schedulable units."""
        return (self.cpu, *self.gpus, *self.npus)
