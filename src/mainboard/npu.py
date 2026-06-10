from __future__ import annotations

from typing import ClassVar

from patos import Registry

from .enums import UnitKind
from .unit import Unit


class NPU(Unit, Registry):
    """Neural processing unit.

    Registry root: concrete vendor providers self-register on import, and
    `all` fans out over them, concatenating each provider's own probe.
    """

    kind: ClassVar[UnitKind] = UnitKind.NPU

    @classmethod
    def all(cls) -> tuple[NPU, ...]:
        """Return NPUs visible across every registered provider."""
        providers = (p for p in cls.registry() if p is not NPU)
        return tuple(npu for provider in providers for npu in provider.all())
