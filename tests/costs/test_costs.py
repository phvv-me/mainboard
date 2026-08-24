from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
from hypothesis import example, given
from hypothesis import strategies as st

from mainboard.costs import (
    BillingModel,
    Catalog,
    Ledger,
    Observation,
    Offer,
    SetupFit,
    catalog_provider,
    from_gpuhunt,
    from_vast,
)

from ..strategies import WORDS

if TYPE_CHECKING:
    from pathlib import Path

# A roster as a catalog really holds one, every offer named in the tame vocabulary so a
# falsifying example stays legible and a case-insensitive query has something to fold.
_ROSTER = st.lists(
    st.builds(Offer, provider=WORDS, gpu=WORDS, rate_usd_hr=st.floats(0.01, 100.0)),
    max_size=6,
)


@given(
    setup=st.floats(0.0, 600.0),
    run=st.floats(0.0, 3600.0),
    drain=st.floats(0.0, 60.0),
    granularity=st.sampled_from([1, 60]),
    minimum=st.sampled_from([0, 60, 120]),
    fees=st.floats(0.0, 5.0),
    rate=st.floats(0.0, 50.0),
)
def test_billing_never_undercharges_wall_time_and_prices_an_hour_at_the_hourly_rate(
    *,
    setup: float,
    run: float,
    drain: float,
    granularity: int,
    minimum: int,
    fees: float,
    rate: float,
) -> None:
    model = BillingModel(
        provider="p",
        rate_usd_hr=rate,
        granularity_s=granularity,
        minimum_s=minimum,
        fees_usd=fees,
    )
    billed = model.billed_seconds(setup_s=setup, run_s=run, drain_s=drain)
    assert billed >= setup + run + drain - 1e-9
    assert billed >= minimum
    assert billed % granularity == 0
    assert model.billed_seconds(setup_s=setup, run_s=run + 60.0, drain_s=drain) >= billed
    assert model.cost_usd(setup_s=setup, run_s=run, drain_s=drain) >= fees
    hourly = BillingModel(provider="p", rate_usd_hr=rate)
    assert hourly.cost_usd(setup_s=0.0, run_s=3600.0) == pytest.approx(rate)


@given(
    submit=st.floats(0.0, 1000.0),
    running=st.floats(0.0, 2000.0),
    ended=st.floats(0.0, 3000.0),
)
@example(submit=100.0, running=0.0, ended=0.0)
@example(submit=100.0, running=104.0, ended=0.0)
@example(submit=100.0, running=104.0, ended=134.0)
def test_an_observation_reads_a_phase_as_zero_until_both_of_its_stamps_land(
    *, submit: float, running: float, ended: float
) -> None:
    observed = Observation(provider="modal", t_submit=submit, t_running=running, t_ended=ended)
    assert observed.setup_s >= 0.0
    assert observed.run_s >= 0.0
    if not running:
        assert (observed.setup_s, observed.run_s) == (0.0, 0.0)
    if not ended:
        assert observed.run_s == 0.0
    if submit <= running <= ended and running and ended:
        assert observed.setup_s + observed.run_s == pytest.approx(ended - submit)


def test_the_ledger_appends_every_observation_and_narrows_by_provider_and_gpu(
    tmp_path: Path,
) -> None:
    ledger = Ledger(tmp_path)
    assert ledger.observations() == []
    recorded = [
        Observation(provider="modal", gpu="H100", t_submit=0.0, t_running=5.0, t_ended=35.0),
        Observation(provider="hpc-ai", gpu="H200", t_submit=0.0, t_running=250.0, t_ended=280.0),
        Observation(provider="modal", gpu="H100", t_submit=0.0, t_running=7.0, t_ended=37.0),
    ]
    for observed in recorded:
        ledger.record(observed)
    assert ledger.observations() == recorded
    assert ledger.observations(provider="modal") == [recorded[0], recorded[2]]
    assert ledger.observations(gpu="H200") == [recorded[1]]
    assert ledger.observations(provider="modal", gpu="H200") == []


@pytest.mark.parametrize(
    ("setups", "samples"),
    [
        pytest.param((4.0, 6.0), None, id="two-observations-are-not-a-fit"),
        pytest.param((5.0, 8.0, 30.0), 3, id="three-observations-fit"),
        pytest.param((5.0, 8.0, 30.0, 0.0), 3, id="a-job-that-never-started-is-not-a-sample"),
    ],
)
def test_a_setup_fit_needs_three_measured_setups_and_brackets_them(
    tmp_path: Path, setups: tuple[float, ...], samples: int | None
) -> None:
    ledger = Ledger(tmp_path)
    for setup in setups:
        ledger.record(
            Observation(provider="modal", t_submit=0.0, t_running=setup, t_ended=setup + 1.0)
        )
    fit = SetupFit.from_ledger(ledger, provider="modal")
    if samples is None:
        assert fit is None
        return
    measured = [setup for setup in setups if setup]
    assert fit is not None
    assert (fit.provider, fit.gpu, fit.samples) == ("modal", "", samples)
    assert min(measured) <= fit.p50_s <= fit.p90_s <= max(measured)
    assert min(measured) <= fit.mean_s <= max(measured)


@given(roster=_ROSTER, gpu=WORDS, provider=WORDS)
def test_a_catalog_narrows_by_hardware_first_and_by_provider_within_it(
    *, roster: list[Offer], gpu: str, provider: str
) -> None:
    catalog = Catalog()
    catalog.add(*roster)
    assert catalog.offers() == catalog.roster == roster
    by_hardware = catalog.offers(gpu=gpu)
    assert by_hardware == catalog.offers(gpu=gpu.upper())
    assert all(offer.gpu == gpu for offer in by_hardware)
    narrowed = catalog.offers(gpu=gpu.upper(), provider=provider)
    assert narrowed == [offer for offer in by_hardware if offer.provider == provider]


def test_quotes_price_every_offer_cheapest_first_and_penalize_the_unmeasured(
    tmp_path: Path,
) -> None:
    ledger = Ledger(tmp_path)
    for _ in range(3):
        ledger.record(
            Observation(provider="modal", gpu="GB200", t_submit=0.0, t_running=2.0, t_ended=3.0)
        )
    measured = Offer(provider="modal", gpu="GB200", rate_usd_hr=11.0)
    unmeasured = Offer(
        provider="hpc-ai", gpu="GB200", rate_usd_hr=7.5, granularity_s=60, minimum_s=60
    )
    catalog = Catalog((measured, unmeasured, Offer(provider="modal", gpu="H200", rate_usd_hr=2.2)))

    quotes = catalog.quotes(gpu="GB200", run_s=30.0, ledger=ledger, default_setup_s=300.0)
    assert [quote.offer for quote in quotes] == [measured, unmeasured]
    assert quotes[0].setup_samples == 3
    assert quotes[0].expected_usd == pytest.approx(
        measured.billing.cost_usd(setup_s=2.0, run_s=30.0)
    )
    assert quotes[1].setup_samples == 0
    assert quotes[1].expected_usd == pytest.approx(
        unmeasured.billing.cost_usd(setup_s=300.0, run_s=30.0)
    )

    unfitted = catalog.quotes(gpu="GB200", run_s=30.0, default_setup_s=300.0)
    assert [quote.setup_samples for quote in unfitted] == [0, 0]
    assert all(quote.expected_usd == pytest.approx(quote.p90_usd) for quote in unfitted)


def test_a_catalog_round_trips_through_ndjson_and_reads_an_absent_file_as_no_offers(
    tmp_path: Path,
) -> None:
    roster = (
        Offer(provider="hpc-ai", gpu="B200-SXM-180GB", rate_usd_hr=3.5, granularity_s=60),
        Offer(provider="modal", gpu="T4", rate_usd_hr=0.59, available=True, source="scraped"),
    )
    target = tmp_path / "feeds" / "catalog.ndjson"
    Catalog(roster).save(target)
    assert Catalog.load(target).roster == list(roster)
    Catalog().save(tmp_path / "empty.ndjson")
    assert Catalog.load(tmp_path / "empty.ndjson").roster == []
    assert Catalog.load(tmp_path / "missing.ndjson").roster == []


def test_an_imported_row_and_a_probed_one_land_under_the_single_name_a_query_asks_for() -> None:
    """gpuhunt spells Vast `vastai` while the live probe and the host kind spell it `vast`, and
    a catalog query narrows by exactly one name, so both feeds are reconciled at this seam."""
    imported = from_gpuhunt(
        [
            SimpleNamespace(
                provider="vastai",
                gpu_name="H100",
                gpu_count=4,
                price=7.6,
                spot=True,
                location="US",
            ),
            SimpleNamespace(
                provider="runpod",
                gpu_name="H100",
                gpu_count=8,
                price=23.92,
                spot=False,
                location=None,
            ),
            SimpleNamespace(
                provider="vastai", gpu_name="X", gpu_count=1, price=None, spot=False, location=""
            ),
        ]
    )
    assert (catalog_provider("vastai"), catalog_provider("runpod")) == ("vast", "runpod")
    assert [offer.provider for offer in imported] == ["vast", "runpod"]
    assert (imported[0].spot, imported[0].gpu_count, imported[0].region) == (True, 4, "US")
    assert imported[1].region == ""
    assert {offer.source for offer in imported} == {"imported:gpuhunt"}
    assert {offer.available for offer in imported} == {None}

    bundle = {
        "gpu_name": "H100",
        "num_gpus": 1,
        "dph_total": 1.8,
        "min_bid": 0.4,
        "rentable": True,
    }
    [on_demand] = from_vast([bundle])
    assert (on_demand.provider, on_demand.source) == ("vast", "probed:vast")
    assert (on_demand.rate_usd_hr, on_demand.available, on_demand.region) == (1.8, True, "")
    [interruptible] = from_vast([{**bundle, "geolocation": "JP"}], spot=True)
    assert (interruptible.rate_usd_hr, interruptible.spot, interruptible.region) == (
        0.4,
        True,
        "JP",
    )
    [unprobed] = from_vast([{"gpu_name": "H100", "num_gpus": 1, "dph_total": 1.8}])
    assert unprobed.available is None

    catalog = Catalog((*imported, on_demand, interruptible))
    assert {offer.source for offer in catalog.offers(provider="vast")} == {
        "imported:gpuhunt",
        "probed:vast",
    }
