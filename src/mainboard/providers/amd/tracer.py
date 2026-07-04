"""AMD annotation backend: ROCTx ranges, visible under `rocprofv3 --marker-trace`."""

from contextlib import suppress
from importlib import import_module
from typing import TYPE_CHECKING, ClassVar

from ...enums import Vendor
from ...profiling.tracer import Tracer

if TYPE_CHECKING:
    from .protocols import Roctx


# `roctx` ships with ROCm (no stubs), so the loader returns the `Roctx` Protocol bound to the
# real module. `import_module` returns an untyped `ModuleType` that pyrefly cannot bridge to a
# Protocol (its `__getattr__` defeats the structural check), hence the `pyrefly: ignore`.
def _load_roctx() -> Roctx:
    return import_module("roctx")  # pyrefly: ignore[bad-return]


roctx: Roctx | None = None
with suppress(ImportError):
    roctx = _load_roctx()


class RoctxTracer(Tracer):
    """ROCTx start/stop ranges and marks; keeps an id stack to model push/pop."""

    vendor: ClassVar[Vendor] = Vendor.AMD
    label: ClassVar[str] = "roctx"

    def __init__(self) -> None:
        self._ids: list[int] = []

    @classmethod
    def is_available(cls) -> bool:
        return roctx is not None

    def push(self, name: str) -> None:
        assert roctx is not None  # built only when ROCTx loaded (see is_available)
        self._ids.append(roctx.rangeStart(name))

    def pop(self) -> None:
        if self._ids:
            assert roctx is not None
            roctx.rangeStop(self._ids.pop())

    def mark(self, name: str) -> None:
        assert roctx is not None
        roctx.mark(name)
