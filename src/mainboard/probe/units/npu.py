import importlib
import logging
from typing import ClassVar

from patos import Registry

from ..enums import UnitKind
from .unit import Unit

logger = logging.getLogger(__name__)


class NPU(Unit, Registry):
    """Neural processing unit, a registry root fanning out like `GPU.all`."""

    kind: ClassVar[UnitKind] = UnitKind.NPU

    @classmethod
    def all(cls) -> tuple[NPU, ...]:
        """NPUs visible across every registered provider."""
        importlib.import_module("mainboard.probe.providers")
        return tuple(npu for provider in cls.implementations() for npu in cls.probe(provider))

    @classmethod
    def probe(cls, provider: type[NPU]) -> tuple[NPU, ...]:
        """One provider's devices, or an empty tuple (logged) when its probe raises."""
        try:
            return tuple(provider.all())
        except Exception:
            logger.warning(
                "NPU provider %s failed to probe, skipping", provider.__name__, exc_info=True
            )
            return ()
