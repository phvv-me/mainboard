"""Structural contracts for the untyped `psutil` result tuples mainboard reads.

`psutil` ships no type stubs, so its `swap_memory()` / `disk_usage()` / `virtual_memory()`
results are otherwise opaque. These Protocols pin the named-tuple fields mainboard reads, so
the models stay typed across the psutil boundary instead of leaking `Any` into their getters.
"""

from typing import Protocol


class SwapMemory(Protocol):
    """The `psutil.swap_memory()` fields mainboard reads: total and used swap bytes."""

    @property
    def total(self) -> int: ...
    @property
    def used(self) -> int: ...


class DiskUsage(Protocol):
    """The `psutil.disk_usage()` fields mainboard reads: total, used, and free bytes."""

    @property
    def total(self) -> int: ...
    @property
    def used(self) -> int: ...
    @property
    def free(self) -> int: ...
