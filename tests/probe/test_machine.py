from typing import TYPE_CHECKING, NoReturn

import pytest

from mainboard import Machine
from mainboard.probe import CPU, GPU, NPU, Scheduler, UnitKind, Vendor

if TYPE_CHECKING:
    from .conftest import FakeNvidiaApis


def test_the_compilers_target_the_newest_cuda_device_on_the_host(
    nvidia_host: FakeNvidiaApis,
) -> None:
    """A native build has to be told which architecture to emit for, and that is read off the
    detected devices rather than configured, paired with the host CPU the flags are tuned to."""
    machine = Machine()
    compilers = machine.compilers
    assert compilers.cuda_arch == "89"
    assert compilers.arch == machine.host.arch
    assert compilers.cpu == machine.host.cpu


def test_the_compilers_refuse_a_host_with_no_cuda_device(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every other subsystem is best effort, and this is the deliberate exception, since a CUDA
    build has no compute capability to target when the machine carries no CUDA device."""
    machine = Machine()
    monkeypatch.setattr(type(machine), "gpus", (GPU(index=0),))
    with pytest.raises(RuntimeError, match="No CUDA device"):
        _ = machine.compilers


def test_the_facade_composes_one_cpu_with_every_detected_gpu_and_npu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`Machine` is one shared view of the host, so it interns a single instance, mirrors the
    host identity into its CPU unit, and lists that CPU ahead of the accelerators it found."""
    machine = Machine()
    assert Machine() is machine

    gpu, npu = GPU(index=0), NPU(index=0)
    monkeypatch.setattr(type(machine), "gpus", (gpu,))
    monkeypatch.setattr(type(machine), "npus", (npu,))
    cpu = machine.cpu
    assert isinstance(cpu, CPU)
    assert cpu.label == machine.host.cpu
    assert cpu.architecture == machine.host.arch
    assert cpu.vendor in Vendor
    assert machine.units == (cpu, gpu, npu)
    assert all(unit.kind in UnitKind for unit in machine.units)
    assert machine.environment.scheduler in Scheduler


def test_the_machine_enumerates_its_providers_and_survives_a_broken_one() -> None:
    """The facade promises best-effort detection rather than a crash, so a provider that throws
    has to be skipped at the `Machine` boundary too and not only inside `GPU.all`."""

    class BrokenGPU(GPU):
        @classmethod
        def all(cls) -> NoReturn:
            raise RuntimeError("backend exploded mid-probe")

    machine = Machine()
    assert not any(isinstance(gpu, BrokenGPU) for gpu in machine.gpus)
    assert isinstance(machine.npus, tuple)
