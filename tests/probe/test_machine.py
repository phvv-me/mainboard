from typing import TYPE_CHECKING

import pytest

from mainboard import Machine
from mainboard.probe import CPU, GPU, NPU, UnitKind, Vendor

if TYPE_CHECKING:
    from .conftest import FakeNvidiaApis


def test_compilers_target_the_newest_cuda_device_on_the_host(nvidia_host: FakeNvidiaApis) -> None:
    """`compilers` pairs the host CPU identity with the detected compute capability."""
    machine = Machine()
    compilers = machine.compilers
    assert compilers.cuda_arch == "89"
    assert compilers.arch == machine.host.arch
    assert compilers.cpu == machine.host.cpu


def test_compilers_refuse_a_host_with_no_cuda_device(monkeypatch: pytest.MonkeyPatch) -> None:
    """A machine whose GPUs are all non-CUDA has no architecture to target, so it raises."""
    machine = Machine()
    monkeypatch.setattr(type(machine), "gpus", (GPU(index=0),))
    machine.__dict__.pop("compilers", None)
    with pytest.raises(RuntimeError, match="No CUDA device"):
        _ = machine.compilers


def test_units_compose_cpu_gpus_and_npus(monkeypatch: pytest.MonkeyPatch) -> None:
    """`units` is the CPU followed by every detected GPU and NPU."""
    machine = Machine()
    gpu, npu = GPU(index=0), NPU(index=0)
    monkeypatch.setattr(type(machine), "gpus", (gpu,))
    monkeypatch.setattr(type(machine), "npus", (npu,))
    machine.__dict__.pop("units", None)
    units = machine.units
    assert units[0] is machine.cpu
    assert units[1:] == (gpu, npu)
    assert all(u.kind in UnitKind for u in units)


def test_cpu_derives_from_host() -> None:
    """The machine CPU mirrors host identity and capacity fields."""
    machine = Machine()
    cpu = machine.cpu
    assert isinstance(cpu, CPU)
    assert cpu.label == machine.host.cpu
    assert cpu.architecture == machine.host.arch
    assert cpu.vendor in Vendor


def test_machine_is_a_singleton() -> None:
    """Two constructions return the exact same instance."""
    assert Machine() is Machine()


def test_gpus_and_npus_default_to_the_registered_providers() -> None:
    """With no fake registered, `gpus`/`npus`/`environment` still resolve without raising."""
    machine = Machine()
    assert isinstance(machine.gpus, tuple)
    assert isinstance(machine.npus, tuple)
    assert machine.environment.scheduler is not None


def test_machine_probe_survives_a_broken_gpu_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """`Machine().gpus` keeps working providers when one backend raises.

    The facade promises best-effort detection rather than a crash, so a
    provider that throws must be skipped at the `Machine` boundary too.
    """

    class BrokenGPU(GPU):
        @classmethod
        def all(cls) -> tuple[GPU, ...]:
            raise RuntimeError("backend exploded mid-probe")

    gpus = Machine().gpus
    assert not any(isinstance(gpu, BrokenGPU) for gpu in gpus)
