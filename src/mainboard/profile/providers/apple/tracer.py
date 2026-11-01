# macOS annotation backend: `os_signpost` intervals, shown in Instruments. The module imports only
# where `os_signpost` does, and the providers package skips it otherwise.

from typing import ClassVar

from os_signpost import Signposter

from ....core.execution import host
from ...tracer import Marker, Tracer, Vendor

_SUBSYSTEM = "me.phvv.mainboard"


class SignpostTracer(Tracer):
    """`os_signpost` intervals."""

    vendor: ClassVar[Vendor] = Vendor.APPLE
    label: ClassVar[str] = "signpost"

    @classmethod
    def is_available(cls) -> bool:
        return host.is_macos()

    def __init__(self) -> None:
        self._signposter = Signposter(_SUBSYSTEM, Signposter.Category.PointsOfInterest)

    def start(self, name: str) -> Marker:
        """Open a correlatable signpost interval and return its exact closer."""
        close = self._signposter.begin_interval(name)
        return lambda: close(None)
