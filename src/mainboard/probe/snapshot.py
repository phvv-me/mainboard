import platform

from patos import FrozenOpenModel

from .enums import Scheduler
from .facts.fabric import Fabric, FabricPort
from .machine import Machine

_SCHEMA_VERSION = 1


class CgroupCap(FrozenOpenModel):
    """The enforced cgroup memory ceiling, or the host total RAM when uncapped."""

    limit_bytes: int = 0
    capped: bool = False


class ScratchInfo(FrozenOpenModel):
    """The host's fastest writable node-local scratch tier."""

    path: str | None = None
    free_bytes: int = 0
    source: str = ""


class GpuFact(FrozenOpenModel):
    """One detected GPU's name, capacity, and dispatch key."""

    name: str
    memory_total_bytes: int
    arch_key: str | None = None


class HostFacts(FrozenOpenModel):
    """A versioned, wire-portable snapshot of one host's compute resources.

    This is the JSON another machine parses (a dispatcher sizing a job against a
    remote host's cgroup cap and scratch space), so it stays a `FrozenOpenModel`,
    letting a reader on an older `schema_version` tolerate a newer writer adding
    fields instead of failing to parse.

    schema_version: format revision, bumped when a field's meaning changes, not
    when one is only added (an addition is what `FrozenOpenModel` already
    tolerates).
    hostname: network name of the probed host.
    cpu_name: CPU model name.
    cpu_logical_cores: logical CPU threads including hyperthreading.
    memory_total_bytes: total system RAM.
    cgroup: the enforced cgroup memory cap, the real OOM-kill ceiling for a job.
    scratch: the fastest writable node-local scratch tier with its free space.
    scheduler: the job scheduler available on the host's PATH.
    gpus: detected GPUs with name, memory capacity, and dispatch key, empty when none.
    fabric: detected InfiniBand/RoCE fabric ports, empty when none.
    """

    schema_version: int = _SCHEMA_VERSION
    hostname: str = ""
    cpu_name: str = ""
    cpu_logical_cores: int = 0
    memory_total_bytes: int = 0
    cgroup: CgroupCap = CgroupCap()
    scratch: ScratchInfo = ScratchInfo()
    scheduler: Scheduler = Scheduler.NONE
    gpus: tuple[GpuFact, ...] = ()
    fabric: tuple[FabricPort, ...] = ()

    @classmethod
    def collected(cls) -> HostFacts:
        """Probe the current host into one serializable snapshot."""
        machine = Machine()
        host = machine.host
        cgroup, scratch = host.cgroup_memory, host.scratch
        return cls(
            hostname=platform.node(),
            cpu_name=machine.cpu.label,
            cpu_logical_cores=host.logical_cpus,
            memory_total_bytes=host.memory.total_bytes,
            cgroup=CgroupCap(limit_bytes=cgroup.limit_bytes, capped=cgroup.capped),
            scratch=ScratchInfo(
                path=str(scratch.path) if scratch.path else None,
                free_bytes=scratch.free_bytes,
                source=scratch.source,
            ),
            scheduler=machine.environment.scheduler,
            gpus=tuple(
                GpuFact(
                    name=gpu.label,
                    memory_total_bytes=gpu.memory.total_bytes,
                    arch_key=gpu.arch_key if gpu.arch_key != "unknown" else None,
                )
                for gpu in machine.gpus
            ),
            fabric=Fabric.probe(),
        )
