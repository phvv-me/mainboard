from .base import FrozenModel


class PcieInfo(FrozenModel):
    """PCIe bus throughput counters sampled by NVML.

    tx_kbs: bytes transmitted from GPU to system memory per second (KB/s).
    rx_kbs: bytes received by GPU from system memory per second (KB/s).
    """

    tx_kbs: int = 0
    rx_kbs: int = 0
