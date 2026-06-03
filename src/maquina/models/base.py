from __future__ import annotations

from functools import cached_property

from pydantic import BaseModel, ConfigDict, Field

_IGNORED_TYPES = (cached_property,)


class Model(BaseModel):
    """Mutable pydantic model for machine schemas."""

    model_config = ConfigDict(ignored_types=_IGNORED_TYPES)


class FrozenModel(BaseModel):
    """Immutable pydantic model for machine schemas."""

    model_config = ConfigDict(
        frozen=True,
        populate_by_name=True,
        ignored_types=_IGNORED_TYPES,
    )


__all__ = ["Field", "FrozenModel", "Model"]
