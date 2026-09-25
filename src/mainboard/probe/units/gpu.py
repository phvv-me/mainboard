import importlib
import logging
from functools import cached_property
from typing import ClassVar

from patos import Registry

from ..enums import UnitKind
from ..facts.memory import Memory
from ..facts.telemetry import Telemetry
from ..facts.utilization import Utilization
from .unit import Unit

logger = logging.getLogger(__name__)


class GPU(Unit, Registry):
    """GPU with static identity, capacity and live sensors.

    A registry root: vendor providers self-register on import and `all` concatenates their
    probes. Identity, capacity, `peak_bandwidth_gbs` and `snapshot` are the whole surface a
    profiler samples a device through, so a discovered GPU needs no adapter to be profiled.
    """

    kind: ClassVar[UnitKind] = UnitKind.GPU

    @cached_property
    def arch_key(self) -> str:
        """A stable, dot-free architecture id for per-arch dispatch.

        Vendors return a precise target such as `sm_90` (NVIDIA) for per-generation config
        tables to key off; the base falls back to the lowercased architecture name.
        """
        return self.architecture.lower()

    @cached_property
    def coherent(self) -> bool:
        """Whether host RAM is a peer of this device's memory over a cache-coherent fabric.

        A provider that can probe the fabric answers for itself; the base knows of none.
        """
        return False

    @cached_property
    def driver(self) -> str:
        """The HOST DRIVER version this device answers under, `610.57.04` shaped, or empty.

        The driver and the CUDA version a driver tops out at are two different facts. Reporting
        the second under the first is how a receipt came to carry `13.3` on a host whose driver
        is `610.57.04`, which is why they are two properties.
        """
        return ""

    @cached_property
    def runtime_version(self) -> tuple[int, int] | None:
        """The compute runtime version as `(major, minor)`, the CUDA one here, when known."""
        return None

    @property
    def memory(self) -> Memory:
        """Current accelerator memory state."""
        return Memory(scope="device", source=self.backend, supported=False)

    @cached_property
    def peak_clock_khz(self) -> int:
        """The highest SM clock the card can run, in kHz; 0 when the backend cannot say."""
        return 0

    @cached_property
    def peak_bandwidth_gbs(self) -> float:
        """Theoretical peak memory bandwidth in GB/s, 0.0 when the provider cannot say."""
        return 0.0

    @property
    def utilization(self) -> Utilization:
        """Current compute and memory-controller utilization."""
        return Utilization()

    @cached_property
    def uuid(self) -> str:
        """Stable GPU identifier when the provider exposes one."""
        return ""

    @classmethod
    def all(cls) -> tuple[GPU, ...]:
        """GPUs visible across every registered provider."""
        importlib.import_module("mainboard.probe.providers")
        return tuple(gpu for provider in cls.implementations() for gpu in cls.probe(provider))

    @classmethod
    def probe(cls, provider: type[GPU]) -> tuple[GPU, ...]:
        """One provider's devices, best effort.

        A provider whose `all` raises (a binding that loads then throws, an unexpected NVML
        error) is logged and skipped, so one broken vendor never sinks the whole machine probe.
        """
        try:
            return tuple(provider.all())
        except Exception:
            logger.warning(
                "GPU provider %s failed to probe, skipping", provider.__name__, exc_info=True
            )
            return ()

    def snapshot(self, name: str = "") -> Telemetry:
        """Point-in-time reading of this GPU's sensors, tagged with region `name`.

        A provider with no sensor access answers with an honest, zeroed reading, never a raise.
        """
        return Telemetry(unit_name=self.label, region=name, utilization=self.utilization)
