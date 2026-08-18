from enum import StrEnum, auto


class DiskKind(StrEnum):
    """Drive technology for a mounted disk or partition."""

    NVME = auto()
    SSD = auto()
    HDD = auto()
    RAMDISK = auto()
    UNKNOWN = auto()
