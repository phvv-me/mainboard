from typing import NoReturn

import pytest

from mainboard.probe import CPU, GPU, NPU, Unit, UnitKind, Vendor


@pytest.mark.parametrize(
    ("unit", "kind", "supported"),
    [
        pytest.param(Unit(), UnitKind.UNKNOWN, True, id="unit"),
        pytest.param(GPU(), UnitKind.GPU, False, id="gpu"),
        pytest.param(NPU(), UnitKind.NPU, True, id="npu"),
    ],
)
def test_a_base_unit_reports_unknown_identity_and_an_empty_memory_reading(
    unit: Unit, kind: UnitKind, supported: bool
) -> None:
    """Every unit root answers with a neutral identity, so a probe never has to guard a field."""
    assert unit.kind is kind
    assert unit.vendor is Vendor.UNKNOWN
    assert unit.backend == "none"
    assert unit.label == "unknown"
    assert unit.architecture == "unknown"
    assert unit.memory.total_bytes == 0
    assert unit.memory.supported is supported


def test_a_base_gpu_reports_zeroed_sensors_and_no_driver() -> None:
    """The GPU root adds sensors a vendor fills in, all reading empty until one does."""
    gpu = GPU(index=0)
    assert gpu.arch_key == "unknown"  # the base key is the lowercased architecture name
    assert gpu.uuid == ""
    assert gpu.driver_version is None
    assert (gpu.utilization.gpu_pct, gpu.utilization.memory_pct) == (0, 0)


@pytest.mark.usefixtures("fake_psutil_memory")
def test_a_cpu_surfaces_its_identity_fields_and_the_system_memory_pool() -> None:
    """A `CPU` renames the host identity it was built from and reads system RAM for capacity."""
    cpu = CPU(
        name_value="Apple M4 Pro",
        architecture_value="arm64",
        logical_cores=14,
        physical_cores=14,
        current_clock_mhz=3200.0,
        vendor=Vendor.APPLE,
    )
    assert cpu.kind is UnitKind.CPU
    assert cpu.backend == "os"
    assert cpu.label == "Apple M4 Pro"
    assert cpu.architecture == "arm64"
    assert cpu.current_clock_mhz == 3200.0
    assert cpu.memory.scope == "system"
    assert cpu.memory.total_bytes == 48 * 1024**3
    assert CPU(name_value="x", architecture_value="x").current_clock_mhz is None


def test_a_provider_that_raises_mid_probe_is_skipped_so_the_others_still_report() -> None:
    """A backend that throws mid-probe never silences the survivors.

    It can load and then throw, an unexpected NVML error or a binding that imports but fails
    on its first call, and the fan-out is best-effort.
    """

    class GoodGPU(GPU):
        @classmethod
        def all(cls) -> tuple[GPU, ...]:
            return (cls(index=0),)

    class BrokenGPU(GPU):
        @classmethod
        def all(cls) -> NoReturn:
            raise RuntimeError("backend exploded mid-probe")

    class GoodNPU(NPU):
        @classmethod
        def all(cls) -> tuple[NPU, ...]:
            return (cls(index=0),)

    class BrokenNPU(NPU):
        @classmethod
        def all(cls) -> NoReturn:
            raise OSError("driver handle vanished")

    gpus, npus = GPU.all(), NPU.all()
    assert any(isinstance(gpu, GoodGPU) for gpu in gpus)
    assert not any(isinstance(gpu, BrokenGPU) for gpu in gpus)
    assert any(isinstance(npu, GoodNPU) for npu in npus)
    assert not any(isinstance(npu, BrokenNPU) for npu in npus)
