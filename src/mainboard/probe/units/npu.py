import importlib
import logging
from typing import ClassVar

from patos import Registry

from ..enums import UnitKind
from .unit import Unit

logger = logging.getLogger(__name__)


class NPU(Unit, Registry):
    """Neural processing unit.

    This is a registry root, so concrete vendor providers self-register on import,
    and `all` fans out over them, concatenating each provider's own probe.
    """

    kind: ClassVar[UnitKind] = UnitKind.NPU

    @classmethod
    def all(cls) -> tuple[NPU, ...]:
        """Return NPUs visible across every registered provider.

        Probing is best-effort per provider, so a backend whose `all` raises is
        logged and skipped so one broken vendor never sinks the whole probe.
        """
        importlib.import_module("mainboard.probe.providers")
        return tuple(npu for provider in cls.implementations() for npu in cls.probe(provider))

    @classmethod
    def probe(cls, provider: type[NPU]) -> tuple[NPU, ...]:
        """One provider's devices, or an empty tuple when its probe fails."""
        try:
            return tuple(provider.all())
        except Exception:
            return cls.skipped(provider)

    @classmethod
    def skipped(cls, provider: type[NPU]) -> tuple[NPU, ...]:
        """Log one provider's failed probe and stand for its absent devices."""
        logger.warning(
            "NPU provider %s failed to probe, skipping", provider.__name__, exc_info=True
        )
        return ()
