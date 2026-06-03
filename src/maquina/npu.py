from __future__ import annotations

from typing import ClassVar

from .enums import UnitKind
from .unit import Unit


class NPU(Unit):
    """Neural processing unit."""

    providers: ClassVar[tuple[type[NPU], ...]] = ()
    kind: ClassVar[UnitKind] = UnitKind.NPU

    @classmethod
    def register_providers(cls, *providers: type[NPU]) -> None:
        """Register concrete providers used by `NPU.all`."""
        cls.providers = providers

    @classmethod
    def all(cls) -> tuple[NPU, ...]:
        """Return NPUs visible to supported providers."""
        return tuple(npu for provider in cls.providers for npu in provider.all())
