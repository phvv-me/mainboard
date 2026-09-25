"""Accepted rental terms and the latest time to begin release."""

from datetime import UTC, datetime, timedelta
from decimal import ROUND_FLOOR, Decimal
from math import isfinite

from patos import FrozenModel
from pydantic import AwareDatetime

from ..core.errors import MissionError
from ..costs.catalog import Offer
from ..runtime.job import walltime_seconds
from .vocabulary import Resources

# Seconds before the budget runs out that release must begin, the time a deletion takes.
_RESERVE_S = 120


class Lease(FrozenModel):
    """One accepted quote, retained before creation, with an absolute release deadline.

    The deadline bounds quoted time charges, not unpriced transfer fees or a provider outage,
    so a live monitor is still required.
    """

    offer: Offer
    release_by: AwareDatetime

    @property
    def expired(self) -> bool:
        """Whether release must take precedence over another probe or evidence transfer."""
        return datetime.now(UTC) >= self.release_by

    @classmethod
    def priced(cls, offer: Offer, resources: Resources, *, setup_s: int = 0) -> Lease:
        """Refuse an unaffordable quote and retain its budget-derived deadline."""
        billing = offer.billing
        values = (billing.rate_usd_hr, billing.fees_usd, resources.max_usd)
        if (
            not all(isfinite(value) for value in values)
            or billing.rate_usd_hr <= 0
            or billing.fees_usd < 0
            or resources.max_usd <= billing.fees_usd
            or billing.granularity_s < 1
            or billing.minimum_s < 0
            or setup_s < 0
        ):
            raise MissionError("rental needs finite, positive, affordable billing terms")
        available = (
            (Decimal(str(resources.max_usd)) - Decimal(str(billing.fees_usd)))
            * 3600
            / Decimal(str(billing.rate_usd_hr))
        )
        quanta = int((available / billing.granularity_s).to_integral_value(ROUND_FLOOR))
        seconds = quanta * billing.granularity_s
        requested = walltime_seconds(resources.walltime) if resources.walltime else 0
        if seconds < billing.minimum_s or seconds <= setup_s + requested + _RESERVE_S:
            raise MissionError(
                "rental quote exceeds budget after setup, walltime, and release reserve"
            )
        lifetime = seconds - _RESERVE_S
        if requested:
            lifetime = min(lifetime, setup_s + requested + _RESERVE_S)
        return cls(offer=offer, release_by=datetime.now(UTC) + timedelta(seconds=lifetime))
