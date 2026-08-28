import importlib
import logging
from functools import cached_property
from typing import ClassVar

from patos import Registry

from ..enums import UnitKind, Vendor
from ..facts.memory import Memory
from ..facts.telemetry import Telemetry
from ..facts.utilization import Utilization
from .unit import Unit

logger = logging.getLogger(__name__)


class GPU(Unit, Registry):
    """GPU with static identity, capacity and live sensors.

    This is a registry root, so concrete vendor providers self-register on import,
    and `all` fans out over them, concatenating each provider's own probe.

    Identity, capacity, `peak_bandwidth_gbs` and `snapshot` together are the whole surface
    a profiler samples a device through, so a discovered GPU can be handed straight to one
    without an adapter standing between them.
    """

    index: int = 0
    kind: ClassVar[UnitKind] = UnitKind.GPU
    vendor: Vendor = Vendor.UNKNOWN
    backend: str = "none"

    @cached_property
    def arch_key(self) -> str:
        """A stable, machine-friendly architecture id for per-arch dispatch.

        Vendor backends return a precise, dot-free target such as `sm_90` (NVIDIA)
        so a per-generation config table can key off it. The base falls back to the
        lowercased human architecture name.
        """
        return self.architecture.lower()

    @cached_property
    def architecture(self) -> str:
        """Human-readable architecture or generation name."""
        return "unknown"

    @cached_property
    def driver(self) -> str:
        """The HOST DRIVER version this device answers under, `610.57.04` shaped, or empty.

        The driver and the CUDA version a driver tops out at are two different facts and only
        one of them is the driver. Reporting the second under the first is how a receipt came to
        carry `13.3` on a host whose driver is `610.57.04`, which is why they are two properties.
        """
        return ""

    @cached_property
    def runtime_version(self) -> tuple[int, int] | None:
        """The compute runtime version as `(major, minor)`, the CUDA one here, when known."""
        return None

    @cached_property
    def label(self) -> str:
        """Human-readable GPU name."""
        return "unknown"

    @property
    def memory(self) -> Memory:
        """Current accelerator memory state."""
        return Memory(scope="device", source=self.backend, supported=False)

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
        """Return GPUs visible across every registered provider.

        Probing is best-effort per provider, so a backend whose `all` raises (a
        binding that loads but then throws, an unexpected NVML error) is logged
        and skipped so one broken vendor never sinks the whole machine probe.
        """
        importlib.import_module("mainboard.probe.providers")
        return tuple(gpu for provider in cls.implementations() for gpu in cls.probe(provider))

    @classmethod
    def probe(cls, provider: type[GPU]) -> tuple[GPU, ...]:
        """One provider's devices, or an empty tuple when its probe fails."""
        try:
            return tuple(provider.all())
        except Exception:
            return cls.skipped(provider)

    @classmethod
    def skipped(cls, provider: type[GPU]) -> tuple[GPU, ...]:
        """Log one provider's failed probe and stand for its absent devices."""
        logger.warning(
            "GPU provider %s failed to probe, skipping", provider.__name__, exc_info=True
        )
        return ()

    def snapshot(self, name: str = "") -> Telemetry:
        """Point-in-time reading of this GPU's sensors, tagged with region `name`.

        The base reports only what every unit already exposes, so a provider with no sensor
        access answers with an honest, zeroed reading rather than raising.
        """
        return Telemetry(unit_name=self.label, region=name, utilization=self.utilization)
