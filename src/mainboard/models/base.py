"""Pydantic v2 base models for mainboard machine schemas.

Defined locally rather than imported from `patos` because the published `patos` package
does not yet expose `Model`/`FrozenModel` (they land in an unreleased version); mirrors
the same `ConfigDict` shape so call sites can switch to the `patos` re-export once that
version ships.
"""

from functools import cached_property

from pydantic import BaseModel, ConfigDict, Field

IGNORED_TYPES: tuple[type, ...] = (cached_property,)


class Model(BaseModel):
    """Mutable pydantic model for machine schemas."""

    model_config = ConfigDict(ignored_types=IGNORED_TYPES)


class FrozenModel(BaseModel):
    """Immutable pydantic model for machine schemas."""

    model_config = ConfigDict(
        frozen=True,
        populate_by_name=True,
        ignored_types=IGNORED_TYPES,
    )


__all__ = ["Field", "FrozenModel", "Model"]
