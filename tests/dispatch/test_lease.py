"""Accepted rental terms survive process boundaries and include deletion time."""

from datetime import UTC, datetime

import pytest

from mainboard import MissionError
from mainboard.costs.catalog import Offer
from mainboard.dispatch.lease import Lease
from mainboard.dispatch.vocabulary import Resources

from .support import created_request


def test_accepted_terms_and_deadline_are_saved_before_the_create_request() -> None:
    allocation = created_request()
    before = datetime.now(UTC)
    offer = Offer(provider="vast", gpu="RTX 5080", rate_usd_hr=1, source="offer:42")
    lease = Lease.priced(offer, Resources(max_usd=2, walltime="00:30:00"), setup_s=3600)
    allocation.begin(lease=lease)
    saved = allocation.cache.creation(allocation.label, allocation.record.target)
    assert saved.verdict == "submitting" and saved.lease == lease
    assert lease.offer.source == "offer:42"
    assert 5519 < (lease.release_by - before).total_seconds() < 5521
    allocation.created("42")
    assert allocation.cache.run("42").lease == lease


@pytest.mark.parametrize("rate", [0.0, -1.0, float("nan"), float("inf")])
def test_unknown_or_invalid_prices_cannot_create_a_lease(rate: float) -> None:
    with pytest.raises(MissionError, match="billing terms"):
        Lease.priced(Offer(provider="vast", gpu="test", rate_usd_hr=rate), Resources(max_usd=2))


def test_the_budget_covers_setup_job_billing_quanta_and_deletion_reserve() -> None:
    offer = Offer(provider="vast", gpu="test", rate_usd_hr=1, granularity_s=3600)
    with pytest.raises(MissionError, match="release reserve"):
        Lease.priced(offer, Resources(max_usd=1.5, walltime="00:30:00"), setup_s=1800)
    before = datetime.now(UTC)
    lease = Lease.priced(offer, Resources(max_usd=1.5))
    assert 3479 < (lease.release_by - before).total_seconds() < 3481


def test_a_minimum_charge_above_the_budget_is_not_treated_as_a_short_rental() -> None:
    with pytest.raises(MissionError, match="release reserve"):
        Lease.priced(
            Offer(provider="vast", gpu="test", rate_usd_hr=1, minimum_s=7200),
            Resources(max_usd=1),
        )
