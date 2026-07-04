from typing import TYPE_CHECKING

from rich import box
from rich.align import Align
from rich.console import Console, RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .enums import UnitKind
from .machine import Machine
from .providers import AppleGPU, AppleNPU, NvidiaGPU

if TYPE_CHECKING:
    from .gpu import GPU
    from .models.memory import Memory
    from .npu import NPU
    from .unit import Unit

_KIND_STYLE = {
    UnitKind.CPU: "bold cyan",
    UnitKind.GPU: "bold magenta",
    UnitKind.NPU: "bold green",
    UnitKind.DSP: "bold yellow",
    UnitKind.MEDIA: "bold blue",
    UnitKind.UNKNOWN: "dim",
}


class MachineView:
    """Simple Rich schematic for the current machine."""

    def __init__(self, machine: Machine | None = None) -> None:
        self.machine: Machine = machine if machine is not None else Machine()

    def print(self, *, color: bool = True) -> None:
        """Render the machine schematic to the terminal."""
        Console(no_color=not color).print(self.renderable())

    def renderable(self) -> RenderableType:
        """Return a compact schematic with connected hardware cells."""
        return Panel(
            self.schematic_grid(),
            title=Text(self.machine.cpu.name, style="bold"),
            subtitle=self.compact_summary(),
            border_style="bright_blue",
            box=box.ROUNDED,
        )

    def schematic_grid(self) -> Table:
        """Render memory on the left and detected units on the right."""
        table = Table.grid(expand=True, padding=(0, 1))
        table.add_column(ratio=5)
        table.add_column(ratio=8)
        table.add_row(
            self.memory_cell(),
            self.connected_units(),
        )
        return table

    def connected_units(self) -> Table:
        """Render detected unit cells with arrows from memory."""
        table = Table.grid(expand=True, padding=(0, 1))
        table.add_column(width=14, justify="center")
        table.add_column(ratio=1)
        for label, connector, cell in self.detected_unit_cells():
            table.add_row(self.arrow_to(label, connector), cell)
        return table

    def detected_unit_cells(self) -> tuple[tuple[str, str, Panel], ...]:
        """Return only units that actually exist on this machine."""
        cells = [("CPU", self.connection_label("cpu"), self.cpu_cell())]
        cells.extend(
            (self.gpu_title(index), self.connection_label("gpu"), self.gpu_cell(index))
            for index in range(len(self.machine.gpus))
        )
        cells.extend(
            (self.npu_title(index), self.connection_label("npu"), self.npu_cell(index))
            for index in range(len(self.machine.npus))
        )
        return tuple(cells)

    def cpu_cell(self) -> Panel:
        """Render the CPU cell."""
        return self.cell(
            "CPU",
            _KIND_STYLE[UnitKind.CPU],
            tuple(self.cpu_rows()),
        )

    def cpu_rows(self) -> list[tuple[str, str]]:
        """Return compact CPU identity and capacity rows."""
        cpu = self.machine.cpu
        rows = [
            ("model", cpu.name),
            ("cores", f"{cpu.physical_cores}C / {cpu.logical_cores}T"),
            ("arch", cpu.architecture),
            ("vendor", cpu.vendor.value),
        ]
        if cpu.current_clock_mhz and cpu.current_clock_mhz >= 100:
            rows.append(("clock", f"{cpu.current_clock_mhz:.0f} MHz"))
        return rows

    def gpu_cell(self, index: int) -> Panel:
        """Render the GPU cell."""
        gpu = self.machine.gpus[index]
        rows = self.gpu_rows(gpu)
        memory = self.distinct_memory(gpu)
        if memory:
            rows.append(("memory", memory))
        return self.cell(self.gpu_title(index), _KIND_STYLE[UnitKind.GPU], tuple(rows))

    def gpu_rows(self, gpu: GPU) -> list[tuple[str, str]]:
        """Return provider-aware GPU rows."""
        rows = [("model", gpu.name), ("backend", gpu.backend), ("arch", gpu.architecture)]
        if isinstance(gpu, AppleGPU):
            if gpu.core_count:
                rows.append(("cores", str(gpu.core_count)))
            if gpu.metal_support:
                rows.append(("metal", self.metal_label(gpu.metal_support)))
        if isinstance(gpu, NvidiaGPU):
            rows.append(("SMs", str(gpu.sm_count)))
            rows.append(("compute", str(gpu.cuda_architecture)))
            rows.append(("clocks", self.gpu_clock_label(gpu)))
            if gpu.peak_bandwidth_gbs:
                rows.append(("bandwidth", f"{gpu.peak_bandwidth_gbs:.0f} GB/s"))
        return rows

    def memory_cell(self) -> Panel:
        """Render the system memory cell."""
        memory = self.machine.host.memory
        return self.cell(
            self.memory_title(),
            _KIND_STYLE[UnitKind.MEDIA],
            (
                ("total", self.bytes(memory.total_bytes)),
                ("used", self.bytes(memory.used_bytes)),
                ("available", self.bytes(memory.free_bytes)),
                ("swap", self.swap_label()),
                ("fabric", self.short_fabric_label()),
                ("units", self.shared_memory_units()),
            ),
        )

    def npu_cell(self, index: int) -> Panel:
        """Render the NPU cell."""
        npu = self.machine.npus[index]
        rows = self.npu_rows(npu)
        memory = self.distinct_memory(npu)
        if memory:
            rows.append(("memory", memory))
        return self.cell(self.npu_title(index), _KIND_STYLE[UnitKind.NPU], tuple(rows))

    def npu_rows(self, npu: NPU) -> list[tuple[str, str]]:
        """Return provider-aware NPU rows."""
        rows = [("model", npu.name), ("backend", npu.backend), ("arch", npu.architecture)]
        if isinstance(npu, AppleNPU):
            rows.append(("api", "Core ML"))
            rows.append(("telemetry", "limited"))
        return rows

    def memory_title(self) -> str:
        """Return the memory cell title."""
        if any(unit.memory.unified for unit in self.machine.units):
            return "Memory (Unified)"
        return "Memory (System)"

    def gpu_title(self, index: int) -> str:
        """Return a compact GPU title."""
        return "GPU" if len(self.machine.gpus) == 1 else f"GPU {index}"

    def npu_title(self, index: int) -> str:
        """Return a compact NPU title."""
        return "NPU" if len(self.machine.npus) == 1 else f"NPU {index}"

    def cell(self, title: str, border_style: str, rows: tuple[tuple[str, str], ...]) -> Panel:
        """Render one schematic cell."""
        table = Table.grid(padding=(0, 1))
        table.add_column(style="dim", no_wrap=True)
        table.add_column(ratio=1)
        for label, value in rows:
            table.add_row(label, value)
        return Panel(table, title=title, border_style=border_style, box=box.ROUNDED)

    def arrow_to(self, label: str, connector: str) -> RenderableType:
        """Render one arrow from memory to a detected unit."""
        text = Text()
        text.append(connector, style="dim")
        text.append("\n=====>\n", style="bright_blue")
        text.append(label, style="bold bright_blue")
        return Align.center(text, vertical="middle")

    def distinct_memory(self, unit: Unit) -> str:
        """Return memory only when it is distinct from the shared system pool."""
        memory = unit.memory
        if memory.unified:
            return ""
        return self.memory_usage(memory)

    def memory_usage(self, memory: Memory) -> str:
        """Return one memory reading in human units."""
        if not memory.supported:
            return "unsupported"
        scope = "unified" if memory.unified else memory.scope
        return f"{self.capacity_pair(memory.used_bytes, memory.total_bytes)} {scope}"

    def fabric_label(self) -> str:
        """Return the dominant host-to-accelerator fabric label."""
        if any(unit.memory.unified for unit in self.machine.units):
            return "unified memory"
        if self.machine.gpus:
            return "PCIe / runtime"
        return "system bus"

    def shared_memory_units(self) -> str:
        """Return unit names sharing or using the memory cell."""
        names = [unit.kind.value.upper() for unit in self.machine.units]
        return ", ".join(names)

    def swap_label(self) -> str:
        """Return compact swap usage."""
        hardware = self.machine.host.memory_hardware
        if hardware.swap_total_bytes == 0:
            return "none"
        return self.capacity_pair(hardware.swap_used_bytes, hardware.swap_total_bytes)

    def gpu_clock_label(self, gpu: NvidiaGPU) -> str:
        """Return compact NVIDIA clock info."""
        clocks = gpu.clocks
        parts = []
        if clocks.sm_mhz:
            parts.append(f"SM {clocks.sm_mhz} MHz")
        if clocks.memory_mhz:
            parts.append(f"mem {clocks.memory_mhz} MHz")
        return ", ".join(parts) if parts else "not exposed"

    def metal_label(self, value: str) -> str:
        """Return a human label for macOS Metal support identifiers."""
        prefix = "spdisplays_metal"
        if value.startswith(prefix):
            return f"Metal {value.removeprefix(prefix)}"
        return value

    def short_fabric_label(self) -> str:
        """Return a short fabric label for the connector."""
        label = self.fabric_label()
        if label == "unified memory":
            return "unified"
        if label == "PCIe / runtime":
            return "PCIe"
        return label

    def connection_label(self, kind: str) -> str:
        """Return the memory connection label for one unit kind."""
        if self.fabric_label() == "unified memory":
            return "unified"
        if kind == "cpu":
            return "system"
        if kind == "gpu":
            return "PCIe"
        return "runtime"

    def compact_summary(self) -> str:
        """Return one short summary line for the schematic."""
        return (
            f"{self.machine.host.arch} | {len(self.machine.units)} units | {self.fabric_label()}"
        )

    def bytes(self, value: int) -> str:
        """Format bytes as a compact binary unit string."""
        amount = float(value)
        for suffix in ("B", "KiB", "MiB", "GiB", "TiB"):
            if amount < 1024 or suffix == "TiB":
                return f"{amount:.1f} {suffix}" if suffix != "B" else f"{int(amount)} B"
            amount /= 1024
        return f"{amount:.1f} TiB"  # pragma: no cover - loop always returns at TiB

    def capacity_pair(self, used_bytes: int, total_bytes: int) -> str:
        """Format used and total bytes with a shared unit when possible."""
        if total_bytes >= 1024**4:
            return f"{used_bytes / 1024**4:.1f} / {total_bytes / 1024**4:.1f} TiB"
        if total_bytes >= 1024**3:
            return f"{used_bytes / 1024**3:.1f} / {total_bytes / 1024**3:.1f} GiB"
        return f"{self.bytes(used_bytes)} / {self.bytes(total_bytes)}"
