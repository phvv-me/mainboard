from __future__ import annotations

import pytest

import maquina
from maquina import CPU, GPU, NPU, Machine, MachineSnapshot
from maquina.enums import UnitKind, Vendor


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


def test_snapshot_aggregates_and_round_trips(monkeypatch: pytest.MonkeyPatch) -> None:
    """`snapshot` builds a CPU/memory/accelerator model that survives JSON."""
    machine = Machine()
    monkeypatch.setattr(type(machine), "gpus", (GPU(index=0),))
    monkeypatch.setattr(type(machine), "npus", (NPU(index=0),))
    snapshot = machine.snapshot()
    assert isinstance(snapshot, MachineSnapshot)
    assert snapshot.cpu.name == machine.cpu.name
    assert snapshot.unit_count == 1 + len(snapshot.gpus) + len(snapshot.npus)
    assert UnitKind.GPU in snapshot.kinds
    assert MachineSnapshot.model_validate_json(snapshot.model_dump_json()) == snapshot


def test_snapshot_tolerates_missing_accelerators(monkeypatch: pytest.MonkeyPatch) -> None:
    """A host with no GPU or NPU yields empty tuples instead of raising."""
    machine = Machine()
    monkeypatch.setattr(type(machine), "gpus", ())
    monkeypatch.setattr(type(machine), "npus", ())
    snapshot = machine.snapshot()
    assert snapshot.gpus == ()
    assert snapshot.npus == ()
    assert snapshot.unit_count == 1
    assert snapshot.kinds == (UnitKind.CPU,)


def test_model_dump_json_round_trips_and_indents(monkeypatch: pytest.MonkeyPatch) -> None:
    """`model_dump_json` rebuilds an equal snapshot and honors indentation."""
    machine = Machine()
    monkeypatch.setattr(type(machine), "gpus", ())
    monkeypatch.setattr(type(machine), "npus", ())
    payload = machine.model_dump_json()
    restored = MachineSnapshot.model_validate_json(payload)
    assert restored.cpu.total_memory_bytes == machine.snapshot().cpu.total_memory_bytes
    assert machine.model_dump_json(indent=2).startswith("{\n")


def test_cpu_derives_from_host(monkeypatch: pytest.MonkeyPatch) -> None:
    """The machine CPU mirrors host identity and capacity fields."""
    machine = Machine()
    cpu = machine.cpu
    assert isinstance(cpu, CPU)
    assert cpu.name == machine.host.cpu
    assert cpu.architecture == machine.host.arch
    assert cpu.vendor in Vendor


class FakeNvidiaGpu(maquina.NvidiaGPU):
    """A fake CUDA GPU exposing a fixed compute capability for compiler tests."""

    @property
    def cuda_architecture(self) -> maquina.ComputeCapability:
        return maquina.ComputeCapability(9, 0)


def test_compilers_require_cuda_devices(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without an NVIDIA GPU the compiler facade refuses to guess CUDA settings."""
    machine = Machine()
    monkeypatch.setattr(type(machine), "gpus", ())
    with pytest.raises(RuntimeError, match="No CUDA devices"):
        _ = machine.compilers


def test_compilers_pick_max_cuda_arch(monkeypatch: pytest.MonkeyPatch) -> None:
    """The compiler facade reads CUDA arch from the highest-capability device."""
    machine = Machine()
    monkeypatch.setattr(type(machine), "gpus", (FakeNvidiaGpu(index=0),))
    compilers = machine.compilers
    assert compilers.cuda_arch == "90"
    assert machine.cuda_architecture == "90"


def test_nvcc_path_reads_compiler_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """`nvcc_path` exposes the absolute path of the detected nvcc compiler."""
    machine = Machine()
    fake_nvcc = type("C", (), {"path": "/usr/local/cuda/bin/nvcc"})()
    fake_compilers = type("SC", (), {"nvcc": fake_nvcc})()
    monkeypatch.setattr(type(machine), "compilers", fake_compilers, raising=False)
    assert machine.nvcc_path == "/usr/local/cuda/bin/nvcc"
