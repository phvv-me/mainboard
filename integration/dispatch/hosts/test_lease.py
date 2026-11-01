"""A rental's accepted terms, and the cap outside mainboard each one records."""

from datetime import UTC, datetime, timedelta
from math import inf

import pytest
from hypothesis import given
from hypothesis import strategies as st

from mainboard import MissionError
from mainboard.costs import Offer
from mainboard.dispatch import Resources
from mainboard.dispatch.hosts.lease import Cap, Lease
from mainboard.enums import Bound

_PREPAID = Cap(provider="vast", bound=Bound.PREPAID)
_THRESHOLD = Cap(provider="shadeform", bound=Bound.THRESHOLD)
_DOLLARS = st.floats(min_value=0.0, max_value=1e6)


@given(limit=_DOLLARS, spent=_DOLLARS, more=_DOLLARS)
def test_a_card_cap_runs_down_with_spend_from_its_limit_and_never_below_zero(
    limit: float, spent: float, more: float
) -> None:
    card = Cap(provider="lambda", bound=Bound.CARD, limit_usd=limit)
    left, later = card.remaining(spent_usd=spent), card.remaining(spent_usd=spent + more)
    assert left is not None and later is not None
    assert 0.0 <= later <= left <= limit


@pytest.mark.parametrize(
    ("cap", "balance", "left"),
    [
        pytest.param(
            Cap(provider="lambda", bound=Bound.CARD, limit_usd=100.0),
            None,
            62.5,
            id="a-card-limit-less-what-the-card-was-charged",
        ),
        pytest.param(_PREPAID, None, None, id="a-prepaid-balance-nobody-read"),
        pytest.param(_PREPAID, 12.3, 12.3, id="a-prepaid-balance-as-the-provider-reports-it"),
        pytest.param(_PREPAID, -2.0, 0.0, id="an-overdrawn-balance-leaves-nothing"),
        pytest.param(_THRESHOLD, 12.3, inf, id="a-threshold-bounds-each-rental-alone"),
    ],
)
def test_a_cap_says_what_its_provider_may_still_bill(
    cap: Cap, balance: float | None, left: float | None
) -> None:
    assert cap.remaining(spent_usd=37.5, balance_usd=balance) == left


def test_a_priced_lease_records_its_budget_and_cap_and_an_older_lease_still_reads() -> None:
    offer = Offer(provider="shadeform", gpu="H100", rate_usd_hr=3.29)
    asked = Resources(max_usd=6.0, walltime="00:30:00")
    lease = Lease.priced(offer, asked, cap=_THRESHOLD, setup_s=3600)
    older = Lease.model_validate_json(lease.model_dump_json(exclude={"max_usd", "cap"}))
    assert Lease.model_validate_json(lease.model_dump_json()) == lease
    assert (lease.max_usd, lease.cap, older.max_usd, older.cap) == (6.0, _THRESHOLD, 0.0, None)
    assert lease.release_by - datetime.now(UTC) <= timedelta(seconds=3600 + 1800 + 120)


def test_a_lease_refuses_a_budget_short_of_the_providers_billed_minimum() -> None:
    held = Offer(provider="hotaisle", gpu="MI300X", rate_usd_hr=3.0, minimum_s=1800)
    with pytest.raises(MissionError, match="exceeds budget"):
        Lease.priced(held, Resources(max_usd=1.0), cap=_PREPAID)


def test_a_lease_whose_budget_buys_the_billed_minimum_runs_to_the_end_of_that_budget() -> None:
    held = Offer(provider="hotaisle", gpu="MI300X", rate_usd_hr=3.0, minimum_s=1800)
    lease = Lease.priced(held, Resources(max_usd=2.0), cap=_PREPAID)
    left = lease.release_by - datetime.now(UTC)
    assert timedelta(seconds=2400 - 120 - 5) < left <= timedelta(seconds=2400 - 120)
