import time

from .base import Field
from .pcie_info import PcieInfo
from .process_info import ProcessInfo  # noqa: TC001 - Pydantic resolves this field at runtime.
from .unit_snapshot import UnitSnapshot


class GPUSnapshot(UnitSnapshot):
    """Point-in-time reading of all GPU sensors.

    pcie: PCIe bus TX/RX throughput counters.
    fan_speed_pct: fan duty cycle as a percentage (0 if no fan or unsupported).
    processes: list of compute processes and their GPU memory usage.
    """

    timestamp_ns: int = Field(default_factory=time.perf_counter_ns)
    pcie: PcieInfo = PcieInfo()
    fan_speed_pct: int = 0
    processes: list[ProcessInfo] = []
