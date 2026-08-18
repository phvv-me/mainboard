from enum import StrEnum, auto


class Vendor(StrEnum):
    """Normalized hardware vendor."""

    ARM = auto()
    APPLE = auto()
    NVIDIA = auto()
    QUALCOMM = auto()
    AMD = auto()
    INTEL = auto()
    UNKNOWN = auto()
