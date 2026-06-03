from __future__ import annotations

import pytest
from rich.console import Console

import mainboard
from mainboard import GPU, NPU, Machine, MachineView
from mainboard.enums import Vendor
from mainboard.providers.apple import AppleGPU, AppleNPU


class _Cpu:
    """Minimal CPU stand-in exposing the fields `cpu_rows` reads."""

    name = "Test CPU"
    physical_cores = 8
    logical_cores = 16
    architecture = "arm64"
    vendor = Vendor.APPLE

    def __init__(self, clock: float) -> None:
        self.current_clock_mhz = clock


def render(view: MachineView) -> str:
    """Render a view to plain text for content assertions."""
    console = Console(no_color=True, width=120, record=True)
    console.print(view.renderable())
    return console.export_text()


def test_renders_bare_host(monkeypatch: pytest.MonkeyPatch) -> None:
    """A host with no accelerators renders just the CPU and memory cells."""
    machine = Machine()
    monkeypatch.setattr(type(machine), "gpus", ())
    monkeypatch.setattr(type(machine), "npus", ())
    view = MachineView(machine)
    text = render(view)
    assert "CPU" in text
    assert "Memory" in text
    assert view.fabric_label() == "system bus"
    assert view.connection_label("gpu") == "PCIe"
    assert view.short_fabric_label() == "system bus"


@pytest.mark.usefixtures("apple_host", "fake_psutil_memory")
def test_renders_apple_unified_machine(monkeypatch: pytest.MonkeyPatch) -> None:
    """An Apple machine renders unified memory plus GPU and NPU cells."""
    machine = Machine()
    monkeypatch.setattr(type(machine), "gpus", tuple(AppleGPU.all()))
    monkeypatch.setattr(type(machine), "npus", tuple(AppleNPU.all()))
    view = MachineView(machine)
    text = render(view)
    assert "GPU" in text
    assert "NPU" in text
    assert "Memory (Unified)" in text
    assert view.fabric_label() == "unified memory"
    assert view.connection_label("gpu") == "unified"
    assert view.short_fabric_label() == "unified"
    assert view.metal_label("spdisplays_metal4") == "Metal 4"
    assert view.metal_label("other") == "other"


@pytest.mark.usefixtures("apple_host", "fake_psutil_memory")
def test_apple_cell_shows_cores_and_metal(monkeypatch: pytest.MonkeyPatch) -> None:
    """Apple GPU and NPU cells include core count, Metal support, and Core ML rows."""
    machine = Machine()
    monkeypatch.setattr(type(machine), "gpus", tuple(AppleGPU.all()))
    monkeypatch.setattr(type(machine), "npus", tuple(AppleNPU.all()))
    text = render(MachineView(machine))
    assert "Metal 4" in text
    assert "20" in text  # GPU core count
    assert "Core ML" in text


def test_nvidia_cell_shows_compute_and_bandwidth(nvidia_host: object) -> None:
    """An NVIDIA GPU cell renders SMs, compute capability, clocks, and bandwidth."""
    gpu = mainboard.NvidiaGPU(index=0)
    machine = Machine()
    view = MachineView(machine)
    rows = view.gpu_rows(gpu)
    labels = dict(rows)
    assert labels["SMs"] == "128"
    assert labels["compute"] == "8.9"
    assert "SM 2520 MHz" in labels["clocks"]
    assert "GB/s" in labels["bandwidth"]
    assert view.gpu_clock_label(gpu).startswith("SM 2520 MHz")


def test_gpu_clock_label_not_exposed_when_zero() -> None:
    """A GPU with zero clocks reports `not exposed` rather than an empty string."""

    class ZeroClockGpu(mainboard.NvidiaGPU):
        @property
        def clocks(self) -> object:
            from mainboard import ClockInfo

            return ClockInfo(sm_mhz=0, memory_mhz=0)

    assert MachineView(Machine()).gpu_clock_label(ZeroClockGpu(index=0)) == "not exposed"


def test_distinct_memory_reports_supported_and_unsupported() -> None:
    """Distinct (non-unified) memory renders capacity, and unsupported reads say so."""
    view = MachineView(Machine())
    bare = GPU(index=0)
    assert view.distinct_memory(bare) == "unsupported"  # zero total → unsupported reading
    distinct = mainboard.MemoryUsage(scope="vram", total_bytes=4 * 1024**3, used_bytes=1024**3)
    assert "vram" in view.memory_usage(distinct)


@pytest.mark.parametrize(
    ("swap_total", "swap_used", "expected"),
    [(0, 0, "none"), (8 * 1024**3, 1024**3, "1.0 / 8.0 GiB")],
)
def test_swap_label(
    swap_total: int, swap_used: int, expected: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Swap renders `none` when absent and a capacity pair when present."""
    machine = Machine()
    memory = type("Mem", (), {"swap_total_bytes": swap_total, "swap_used_bytes": swap_used})()
    monkeypatch.setattr(
        type(machine), "host", type("Host", (), {"memory": memory})(), raising=False
    )
    assert MachineView(machine).swap_label() == expected


def test_pcie_fabric_when_discrete_gpu_present(monkeypatch: pytest.MonkeyPatch) -> None:
    """A discrete GPU with no unified memory reports a PCIe/runtime fabric."""
    machine = Machine()
    monkeypatch.setattr(type(machine), "gpus", (GPU(index=0),))
    monkeypatch.setattr(type(machine), "npus", ())
    view = MachineView(machine)
    assert view.fabric_label() == "PCIe / runtime"
    assert view.short_fabric_label() == "PCIe"
    assert view.connection_label("npu") == "runtime"


def test_print_runs_without_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """The top-level `print` path renders to the terminal without raising."""
    machine = Machine()
    monkeypatch.setattr(type(machine), "gpus", ())
    monkeypatch.setattr(type(machine), "npus", ())
    MachineView(machine).print(color=False)


@pytest.mark.parametrize(
    ("value", "expected"),
    [(0, "0 B"), (2048, "2.0 KiB"), (5 * 1024**3, "5.0 GiB"), (3 * 1024**4, "3.0 TiB")],
)
def test_bytes_formatting(value: int, expected: str) -> None:
    """Byte formatting picks a compact binary unit per magnitude."""
    assert MachineView(Machine()).bytes(value) == expected


def test_capacity_pair_shares_units() -> None:
    """A used/total pair shares the largest fitting binary unit."""
    view = MachineView(Machine())
    assert view.capacity_pair(1024**3, 4 * 1024**3) == "1.0 / 4.0 GiB"
    assert view.capacity_pair(1024**4, 2 * 1024**4) == "1.0 / 2.0 TiB"
    assert view.capacity_pair(512, 1024) == "512 B / 1.0 KiB"


def test_cpu_cell_shows_clock_when_high_enough(monkeypatch: pytest.MonkeyPatch) -> None:
    """A CPU clock at or above 100 MHz is rendered, lower readings are omitted."""
    machine = Machine()
    monkeypatch.setattr(type(machine), "gpus", ())
    monkeypatch.setattr(type(machine), "npus", ())
    view = MachineView(machine)
    monkeypatch.setattr(type(machine), "cpu", _Cpu(3200.0), raising=False)
    assert ("clock", "3200 MHz") in view.cpu_rows()
    monkeypatch.setattr(type(machine), "cpu", _Cpu(40.0), raising=False)
    assert all(label != "clock" for label, _ in view.cpu_rows())


def test_apple_gpu_rows_omit_missing_cores_and_metal() -> None:
    """An Apple GPU with no core count or Metal string omits those rows."""

    class BareAppleGpu(AppleGPU):
        @property
        def core_count(self) -> int:
            return 0

        @property
        def metal_support(self) -> str:
            return ""

    rows = MachineView(Machine()).gpu_rows(BareAppleGpu(index=0))
    labels = {label for label, _ in rows}
    assert "cores" not in labels
    assert "metal" not in labels


def test_nvidia_gpu_rows_omit_zero_bandwidth(nvidia_host: object) -> None:
    """An NVIDIA GPU with zero peak bandwidth omits the bandwidth row."""

    class NoBandwidthGpu(mainboard.NvidiaGPU):
        @property
        def peak_bandwidth_gbs(self) -> float:
            return 0.0

    rows = MachineView(Machine()).gpu_rows(NoBandwidthGpu(index=0))
    assert all(label != "bandwidth" for label, _ in rows)


def test_gpu_and_npu_cells_render_distinct_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    """GPU and NPU cells append a distinct (non-unified) memory row when present."""

    class VramGpu(GPU):
        @property
        def memory_readings(self) -> tuple[mainboard.MemoryUsage, ...]:
            return (
                mainboard.MemoryUsage(scope="vram", total_bytes=8 * 1024**3, used_bytes=1024**3),
            )

    class VramNpu(NPU):
        @property
        def memory_readings(self) -> tuple[mainboard.MemoryUsage, ...]:
            return (mainboard.MemoryUsage(scope="sram", total_bytes=1024**3, used_bytes=512),)

    machine = Machine()
    monkeypatch.setattr(type(machine), "gpus", (VramGpu(index=0),))
    monkeypatch.setattr(type(machine), "npus", (VramNpu(index=0),))
    text = render(MachineView(machine))
    assert "vram" in text
    assert "sram" in text


def test_distinct_memory_empty_readings_is_blank() -> None:
    """A unit with no memory readings contributes no distinct-memory string."""

    class NoMemUnit(GPU):
        @property
        def memory_readings(self) -> tuple[mainboard.MemoryUsage, ...]:
            return ()

    assert MachineView(Machine()).distinct_memory(NoMemUnit(index=0)) == ""


def test_bytes_caps_at_tebibytes() -> None:
    """The byte formatter saturates at tebibytes for very large values."""
    assert MachineView(Machine()).bytes(5 * 1024**5) == "5120.0 TiB"


def test_default_view_uses_singleton_machine() -> None:
    """Constructing a view with no machine falls back to the singleton."""
    view = MachineView()
    assert isinstance(view.machine, Machine)
