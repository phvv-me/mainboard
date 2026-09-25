import psutil
from patos import FrozenModel


class Memory(FrozenModel):
    """Memory usage for a host, unit, or memory region.

    used_bytes, free_bytes: 0 when unknown.
    scope: region name, e.g. `system`, `vram`, `unified`.
    unified: whether CPU and accelerator share the memory pool.
    source: provider that produced the value.
    supported: whether this platform exposes the reading.
    """

    total_bytes: int = 0
    used_bytes: int = 0
    free_bytes: int = 0
    scope: str = ""
    unified: bool = False
    source: str = ""
    supported: bool = True

    @property
    def free_gb(self) -> float:
        """Free capacity in gibibytes."""
        return self.free_bytes / 1024**3

    @property
    def percent_used(self) -> float:
        """Percentage of total memory currently used, or 0 when total is 0."""
        return self.used_bytes / self.total_bytes * 100 if self.total_bytes else 0.0

    @property
    def total_gb(self) -> float:
        """Total capacity in gibibytes."""
        return self.total_bytes / 1024**3

    @property
    def used_gb(self) -> float:
        """Used capacity in gibibytes."""
        return self.used_bytes / 1024**3

    @classmethod
    def system(cls, scope: str = "system", *, unified: bool = False) -> Memory:
        """Live system RAM usage sampled from psutil, free being what is available."""
        ram = psutil.virtual_memory()
        return cls(
            scope=scope,
            total_bytes=ram.total,
            used_bytes=ram.used,
            free_bytes=ram.available,
            unified=unified,
            source="psutil",
        )
