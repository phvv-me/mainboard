from __future__ import annotations

import time

from .base import Field
from .clock_info import ClockInfo
from .mem_info import MemInfo
from .pcie_info import PcieInfo
from .process_info import ProcessInfo  # noqa: TC001 - Pydantic resolves this field at runtime.
from .unit_snapshot import UnitSnapshot


class GPUSnapshot(UnitSnapshot):
    """Point-in-time reading of all GPU sensors.

    gpu_memory: GPU memory allocation state.
    gpu_clocks: current SM and memory clock frequencies.
    pcie: PCIe bus TX/RX throughput counters.
    fan_speed_pct: fan duty cycle as a percentage (0 if no fan or unsupported).
    processes: list of compute processes and their GPU memory usage.
    """

    timestamp_ns: int = Field(default_factory=time.perf_counter_ns)
    gpu_memory: MemInfo = MemInfo()
    gpu_clocks: ClockInfo = ClockInfo()
    pcie: PcieInfo = PcieInfo()
    fan_speed_pct: int = 0
    processes: list[ProcessInfo] = []
