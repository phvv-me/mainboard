"""Pydantic v2 base models for mainboard machine schemas.

Defined locally rather than imported from `patos` because the published `patos` package
does not yet expose `Model`/`FrozenModel` (they land in an unreleased version); mirrors
the same `ConfigDict` shape so call sites can switch to the `patos` re-export once that
version ships.
"""

from collections.abc import Sequence
from functools import cached_property
from typing import Annotated

from pydantic import AfterValidator, BaseModel, ConfigDict, Field

IGNORED_TYPES: tuple[type, ...] = (cached_property,)


def _frozen[Item](values: Sequence[Item]) -> tuple[Item, ...]:
    """Normalize one accepted sequence to stable immutable storage."""
    return tuple(values)


type FrozenSequence[Item] = Annotated[Sequence[Item], AfterValidator(_frozen)]


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


__all__ = ["Field", "FrozenModel", "FrozenSequence", "Model"]
