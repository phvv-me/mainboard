"""Pydantic v2 base models for mainboard machine schemas.

The house bases live in :mod:`patos`; mainboard re-exports the two its schemas use plus `Field`,
so call sites import everything from one place.
"""

from patos import FrozenModel, Model
from pydantic import Field

__all__ = ["Field", "FrozenModel", "Model"]
