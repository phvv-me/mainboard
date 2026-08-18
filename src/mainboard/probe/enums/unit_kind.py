from enum import StrEnum, auto


class UnitKind(StrEnum):
    """Schedulable execution-resource category."""

    CPU = auto()
    GPU = auto()
    NPU = auto()
    DSP = auto()
    MEDIA = auto()
    UNKNOWN = auto()
