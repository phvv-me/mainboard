import importlib
import logging
from functools import cached_property
from typing import ClassVar

from patos import Registry

from ..enums import UnitKind, Vendor
from ..facts.memory import Memory
from ..facts.utilization import Utilization
from .unit import Unit

logger = logging.getLogger(__name__)


class GPU(Unit, Registry):
    """GPU with static identity and capacity.

    This is a registry root, so concrete vendor providers self-register on import,
    and `all` fans out over them, concatenating each provider's own probe.
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
    def driver_version(self) -> tuple[int, int] | None:
        """Driver or runtime version as `(major, minor)` when known."""
        return None

    @cached_property
    def label(self) -> str:
        """Human-readable GPU name."""
        return "unknown"

    @property
    def memory(self) -> Memory:
        """Current accelerator memory state."""
        return Memory(scope="device", source=self.backend, supported=False)

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
