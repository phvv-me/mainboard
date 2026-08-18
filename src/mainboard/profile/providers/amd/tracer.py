# AMD annotation backend: ROCTx ranges, visible under `rocprofv3 --marker-trace`.

from contextlib import suppress
from importlib import import_module
from typing import TYPE_CHECKING, ClassVar, cast

from ...tracer import Marker, Tracer, Vendor

if TYPE_CHECKING:
    from .protocols import Roctx

roctx: Roctx | None = None
with suppress(ImportError):
    roctx = cast("Roctx", import_module("roctx"))


class RoctxTracer(Tracer):
    """ROCTx start/stop ranges and marks; keeps an id stack to model push/pop."""

    vendor: ClassVar[Vendor] = Vendor.AMD
    label: ClassVar[str] = "roctx"

    def __init__(self) -> None:
        self._ids: list[int] = []

    @classmethod
    def is_available(cls) -> bool:
        return roctx is not None

    def mark(self, name: str) -> None:
        assert roctx is not None  # noqa: S101  reason=see push() since=2026-08-16
        roctx.mark(name)

    def pop(self) -> None:
        if self._ids:
            assert roctx is not None  # noqa: S101  reason=see push() since=2026-08-16
            roctx.rangeStop(self._ids.pop())

    def push(self, name: str) -> None:
        assert roctx is not None  # noqa: S101  reason=built only when ROCTx loaded (see is_available) since=2026-08-16
        self._ids.append(roctx.rangeStart(name))

    def start(self, name: str) -> Marker:
        """Open a correlatable range and return its exact closer."""
        api = roctx
        if api is None:
            return lambda: None
        range_id = api.rangeStart(name)
        return lambda: api.rangeStop(range_id)
