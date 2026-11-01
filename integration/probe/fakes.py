from collections import Counter
from collections.abc import Callable
from time import sleep

from mainboard.probe.stress import TorchKernels

from .constants import OPERATION_SECONDS, UNSUPPORTED

_CLOCK_KHZ = 1_000_000


class CountedKernels(TorchKernels):
    """A device whose every operation takes `OPERATION_SECONDS` and is counted by its name."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: Counter[str] = Counter()

    def _operation(self, name: str) -> Callable[[], None]:
        def operate() -> None:
            self.calls[name] += 1
            if name == UNSUPPORTED:
                raise RuntimeError("no such precision\nin this device")
            sleep(OPERATION_SECONDS)

        return operate

    def copy(self, path: str, megabytes: int) -> Callable[[], None]:
        return self._operation(path)

    def describe(self) -> tuple[str, tuple[int, int], int, int]:
        return "Counted", (9, 0), 100, _CLOCK_KHZ

    def gemm(self, precision: str, n: int) -> Callable[[], None]:
        return self._operation(precision)

    def synchronize(self) -> None:
        self.calls["synchronize"] += 1
