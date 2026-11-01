"""Accepted rental terms, the latest time to begin release, and the cap outside this process."""

from datetime import UTC, datetime, timedelta
from decimal import ROUND_FLOOR, Decimal
from math import inf, isfinite

from patos import FrozenModel
from pydantic import AwareDatetime, Field

from ...core.constants.units import RELEASE_SECONDS, SECONDS_PER_HOUR
from ...core.errors import MissionError
from ...costs.catalog import Offer
from ...costs.model import BillingModel
from ...enums import Bound
from ...runtime.job import walltime_seconds
from ..jobs.vocabulary import Resources


class Cap(FrozenModel):
    """The bound outside this process on what one provider can bill, recorded on every lease.

    provider: the backend kind it bounds.
    limit_usd: the declared card limit of a `CARD` bound; 0 for the others, whose figure the
        provider holds (its balance, the threshold each rental is sent).
    """

    provider: str
    bound: Bound
    limit_usd: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)

    def remaining(
        self, *, spent_usd: float = 0.0, balance_usd: float | None = None
    ) -> float | None:
        """What this cap still lets its provider bill, None while its balance is unread.

        Pure, so a policy checks a rent against the cap the rent was admitted under. A threshold
        bounds each rental at its own budget and leaves the account nothing to run down (`inf`).

        spent_usd: what the card was charged in its current period; only a card reads it.
        balance_usd: the balance a prepaid provider reports (`Standing.credit_usd`).
        """
        match self.bound:
            case Bound.CARD:
                return max(self.limit_usd - spent_usd, 0.0)
            case Bound.PREPAID:
                return None if balance_usd is None else max(balance_usd, 0.0)
            case Bound.THRESHOLD:
                return inf


class Lease(FrozenModel):
    """One accepted quote, retained before creation, with an absolute release deadline.

    The deadline bounds quoted time charges, not unpriced transfer fees or a provider outage,
    so a live monitor is still required, and `cap` names what bounds the bill without one.

    max_usd: the budget the quote was accepted under, the spend a `THRESHOLD` provider is sent.
    cap: None, with `max_usd` 0, only on a lease recorded before caps were (2026-10-29), which
        the run registry still reads.
    """

    offer: Offer
    release_by: AwareDatetime
    max_usd: float = 0.0
    cap: Cap | None = None

    @classmethod
    def priced(cls, offer: Offer, resources: Resources, *, cap: Cap, setup_s: int = 0) -> Lease:
        """Refuse an unaffordable quote and retain its budget-derived deadline under `cap`."""
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
        seconds = cls._billable_seconds(billing, resources.max_usd)
        requested = walltime_seconds(resources.walltime) if resources.walltime else 0
        if seconds < billing.minimum_s or seconds <= setup_s + requested + RELEASE_SECONDS:
            raise MissionError(
                "rental quote exceeds budget after setup, walltime, and release reserve"
            )
        lifetime = seconds - RELEASE_SECONDS
        if requested:
            lifetime = min(lifetime, setup_s + requested + RELEASE_SECONDS)
        return cls(
            offer=offer,
            release_by=datetime.now(UTC) + timedelta(seconds=lifetime),
            max_usd=resources.max_usd,
            cap=cap,
        )

    @property
    def expired(self) -> bool:
        """Whether release must take precedence over another probe or evidence transfer."""
        return datetime.now(UTC) >= self.release_by

    @staticmethod
    def _billable_seconds(billing: BillingModel, max_usd: float) -> int:
        """Whole billing quanta that `max_usd` buys once the fees are paid, in seconds."""
        available = (
            (Decimal(str(max_usd)) - Decimal(str(billing.fees_usd)))
            * SECONDS_PER_HOUR
            / Decimal(str(billing.rate_usd_hr))
        )
        quanta = int((available / billing.granularity_s).to_integral_value(ROUND_FLOOR))
        return quanta * billing.granularity_s
