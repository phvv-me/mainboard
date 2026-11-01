# AMD annotation backend: ROCTx ranges, visible under `rocprofv3 --marker-trace`. The module
# imports only where `roctx` does, and the providers package skips it otherwise.

from typing import ClassVar

import roctx

from ...tracer import Marker, Tracer, Vendor


class RoctxTracer(Tracer):
    """ROCTx start/stop ranges."""

    vendor: ClassVar[Vendor] = Vendor.AMD
    label: ClassVar[str] = "roctx"

    @classmethod
    def is_available(cls) -> bool:
        return True

    def start(self, name: str) -> Marker:
        """Open a correlatable range and return its exact closer."""
        range_id = roctx.rangeStart(name)
        return lambda: roctx.rangeStop(range_id)
