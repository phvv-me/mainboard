from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

import maquina


def test_cpu_exposes_kind_and_vendor() -> None:
    """CPU exposes neutral scheduling fields directly."""
    unit = maquina.CPU(
        name_value="Apple M4 Pro",
        architecture_value="arm64",
        vendor=maquina.Vendor.APPLE,
    )

    assert unit.kind == maquina.UnitKind.CPU
    assert unit.vendor == maquina.Vendor.APPLE
    assert unit.name == "Apple M4 Pro"
    assert unit.architecture == "arm64"


def test_gpu_exposes_kind_and_vendor() -> None:
    """GPU exposes neutral scheduling fields directly."""
    unit = maquina.GPU()

    assert unit.kind == maquina.UnitKind.GPU
    assert unit.vendor == maquina.Vendor.UNKNOWN


def test_npu_exposes_kind_and_vendor() -> None:
    """NPU exposes neutral scheduling fields directly."""
    unit = maquina.NPU()

    assert unit.kind == maquina.UnitKind.NPU
    assert unit.vendor == maquina.Vendor.UNKNOWN


def test_provider_gpus_expose_kind_and_vendor_without_backend_imports() -> None:
    """Provider identity fields do not require touching CUDA or system_profiler."""
    apple = maquina.AppleGPU(index=0)
    neural = maquina.AppleNPU(index=0)
    nvidia = maquina.NvidiaGPU(index=0)
    amd = maquina.AMDGPU(index=0)
    intel = maquina.IntelGPU(index=0)
    qualcomm = maquina.QualcommGPU(index=0)

    assert apple.kind == maquina.UnitKind.GPU
    assert apple.vendor == maquina.Vendor.APPLE
    assert neural.kind == maquina.UnitKind.NPU
    assert neural.vendor == maquina.Vendor.APPLE
    assert nvidia.kind == maquina.UnitKind.GPU
    assert nvidia.vendor == maquina.Vendor.NVIDIA
    assert amd.vendor == maquina.Vendor.AMD
    assert intel.vendor == maquina.Vendor.INTEL
    assert qualcomm.vendor == maquina.Vendor.QUALCOMM


def test_future_provider_stubs_are_ci_safe() -> None:
    """Future providers are importable and inert until implemented."""
    providers = (
        maquina.AMDGPU,
        maquina.IntelGPU,
        maquina.IntelNPU,
        maquina.QualcommGPU,
        maquina.QualcommNPU,
    )

    assert all(not provider.is_available() for provider in providers)
    assert all(provider.all() == () for provider in providers)


def test_detected_machine_units_expose_kind_and_vendor() -> None:
    """Detected units expose neutral scheduling fields directly."""
    units = maquina.Machine().units

    assert units
    assert all(unit.kind in maquina.UnitKind for unit in units)
    assert all(unit.vendor in maquina.Vendor for unit in units)


def test_cpu_snapshot_uses_neutral_unit_shape() -> None:
    """CPU snapshots expose the shared `UnitSnapshot` API."""
    unit = maquina.CPU(
        name_value="Apple M4 Pro",
        architecture_value="arm64",
        logical_cores=14,
        physical_cores=14,
        total_memory_value_bytes=48 * 1024**3,
        vendor=maquina.Vendor.APPLE,
    )

    snapshot = unit.snapshot("region")

    assert isinstance(snapshot, maquina.UnitSnapshot)
    assert snapshot.name == "region"
    assert snapshot.unit_name == "Apple M4 Pro"
    assert snapshot.kind == maquina.UnitKind.CPU
    assert snapshot.vendor == maquina.Vendor.APPLE
    assert snapshot.memory


def test_gpu_snapshot_extends_neutral_unit_shape() -> None:
    """GPU snapshots preserve neutral fields and add GPU telemetry."""
    unit = maquina.GPU()

    snapshot = unit.snapshot("kernel")

    assert isinstance(snapshot, maquina.GPUSnapshot)
    assert isinstance(snapshot, maquina.UnitSnapshot)
    assert snapshot.name == "kernel"
    assert snapshot.kind == maquina.UnitKind.GPU
    assert snapshot.gpu_memory.total_bytes == 0


def test_machine_snapshot_probes_the_host() -> None:
    """`Machine.snapshot` aggregates CPU, memory, GPUs, and NPUs."""
    snapshot = maquina.Machine().snapshot()

    assert isinstance(snapshot, maquina.MachineSnapshot)
    assert snapshot.cpu.name
    assert snapshot.cpu.architecture
    assert snapshot.cpu.logical_cores > 0
    assert snapshot.memory.total_bytes > 0
    assert snapshot.unit_count == 1 + len(snapshot.gpus) + len(snapshot.npus)
    assert maquina.UnitKind.CPU in snapshot.kinds
    assert all(isinstance(gpu, maquina.GPUSnapshot) for gpu in snapshot.gpus)
    assert all(isinstance(npu, maquina.UnitSnapshot) for npu in snapshot.npus)


def test_machine_model_dump_json_round_trips() -> None:
    """`Machine.model_dump_json` yields JSON that rebuilds an equal snapshot."""
    machine = maquina.Machine()

    payload = machine.model_dump_json()
    restored = maquina.MachineSnapshot.model_validate_json(payload)

    assert restored.cpu.total_memory_bytes == machine.snapshot().cpu.total_memory_bytes
    assert len(restored.gpus) == len(machine.gpus)
    assert maquina.Machine().model_dump_json(indent=2).startswith("{\n")


def test_machine_snapshot_tolerates_missing_accelerators() -> None:
    """A host with no GPU or NPU yields empty tuples, not an error."""
    machine = maquina.Machine()
    original_gpu = type(machine).gpus
    original_npu = type(machine).npus
    machine.__dict__["gpus"] = ()
    machine.__dict__["npus"] = ()
    try:
        snapshot = machine.snapshot()
        assert snapshot.gpus == ()
        assert snapshot.npus == ()
        assert snapshot.unit_count == 1
        assert snapshot.kinds == (maquina.UnitKind.CPU,)
    finally:
        machine.__dict__.pop("gpus", None)
        machine.__dict__.pop("npus", None)
        assert type(machine).gpus is original_gpu
        assert type(machine).npus is original_npu


def test_compute_capability_string_api() -> None:
    """Compute capability has stable human and debug strings."""
    capability = maquina.ComputeCapability(12, 1)

    assert str(capability) == "12.1"
    assert repr(capability) == "ComputeCapability(12, 1)"


@given(
    total=st.integers(min_value=1, max_value=10**15),
    used=st.integers(min_value=0, max_value=10**15),
)
def test_memory_usage_utilization_is_a_percentage(total: int, used: int) -> None:
    """Memory usage percentages match used divided by total."""
    used = min(used, total)

    usage = maquina.MemoryUsage(
        scope="test",
        total_bytes=total,
        used_bytes=used,
        free_bytes=total - used,
    )

    assert usage.utilization_pct == used / total * 100
    assert usage.total_gb == total / 1024**3
    assert usage.used_gb == used / 1024**3
    assert usage.free_gb == (total - used) / 1024**3
    assert maquina.MemoryUsage(scope="empty").utilization_pct == 0.0


@given(
    total=st.integers(min_value=0, max_value=10**15),
    used=st.integers(min_value=0, max_value=10**15),
)
def test_gpu_memory_utilization_handles_empty_and_nonempty_memory(total: int, used: int) -> None:
    """GPU memory utilization is zero for empty devices and proportional otherwise."""
    used = min(used, total)

    memory = maquina.MemInfo(total_bytes=total, used_bytes=used, free_bytes=total - used)

    if total == 0:
        assert memory.utilization_pct == 0.0
    else:
        assert memory.utilization_pct == used / total * 100
    assert memory.total_mb == total / 1024**2
    assert memory.used_mb == used / 1024**2
    assert memory.free_mb == (total - used) / 1024**2
