from functools import cached_property
from typing import ClassVar

from patos import FrozenModel

from ..enums import UnitKind, Vendor
from ..facts.memory import Memory


class Unit(FrozenModel):
    """Schedulable hardware execution resource.

    A unit can be a CPU package or cluster, GPU, NPU, DSP, or other hardware
    engine that executes work over memory.
    """

    index: int = 0
    kind: ClassVar[UnitKind] = UnitKind.UNKNOWN
    vendor: Vendor = Vendor.UNKNOWN
    backend: str = "none"

    @cached_property
    def architecture(self) -> str:
        """Human-readable architecture or generation."""
        return "unknown"

    @cached_property
    def label(self) -> str:
        """Human-readable unit name."""
        return "unknown"

    @property
    def memory(self) -> Memory:
        """Memory visible to this unit."""
        return Memory()
