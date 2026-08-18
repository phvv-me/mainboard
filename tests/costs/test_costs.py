from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
from hypothesis import given
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

if TYPE_CHECKING:
    from pathlib import Path

_MODAL = BillingModel(provider="modal", gpu="H100", rate_usd_hr=4.56, granularity_s=1)
_HPCAI = BillingModel(
    provider="hpc-ai", gpu="H200", rate_usd_hr=2.20, granularity_s=60, minimum_s=60
)


def test_per_second_vs_per_minute_billing_for_a_short_job() -> None:
    assert _MODAL.billed_seconds(setup_s=4, run_s=30) == 34
    assert _HPCAI.billed_seconds(setup_s=240, run_s=30) == 300
    assert _MODAL.cost_usd(setup_s=4, run_s=30) == pytest.approx(4.56 * 34 / 3600)
    assert _HPCAI.cost_usd(setup_s=240, run_s=30) == pytest.approx(2.20 * 300 / 3600)


def test_minimum_floor_and_fees_apply() -> None:
    floor = BillingModel(provider="x", rate_usd_hr=3.6, minimum_s=120, fees_usd=0.05)
    assert floor.billed_seconds(setup_s=0, run_s=1) == 120
    assert floor.cost_usd(setup_s=0, run_s=1) == pytest.approx(0.05 + 0.12)


@given(
    setup=st.floats(0, 600),
    run=st.floats(0, 3600),
    drain=st.floats(0, 60),
    granularity=st.sampled_from([1, 60]),
)
def test_billed_seconds_never_undercharge_wall_time(
    *, setup: float, run: float, drain: float, granularity: int
) -> None:
    model = BillingModel(provider="p", rate_usd_hr=1.0, granularity_s=granularity)
    billed = model.billed_seconds(setup_s=setup, run_s=run, drain_s=drain)
    assert billed >= setup + run + drain - 1e-9
    assert billed % granularity == 0


def test_observation_derives_phases_and_tolerates_missing() -> None:
    complete = Observation(provider="modal", t_submit=100.0, t_running=104.0, t_ended=134.0)
    assert complete.setup_s == pytest.approx(4.0)
    assert complete.run_s == pytest.approx(30.0)
    unstarted = Observation(provider="modal", t_submit=100.0)
    assert unstarted.setup_s == pytest.approx(0.0)
    assert unstarted.run_s == pytest.approx(0.0)
    unfinished = Observation(provider="modal", t_submit=100.0, t_running=104.0)
    assert unfinished.run_s == pytest.approx(0.0)


def test_ledger_round_trips_and_filters(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path)
    assert ledger.observations() == []
    for setup, provider, gpu in (
        (5, "modal", "H100"),
        (250, "hpc-ai", "H200"),
        (7, "modal", "H100"),
    ):
        ledger.record(
            Observation(
                provider=provider,
                gpu=gpu,
                t_submit=0.0,
                t_running=float(setup),
                t_ended=setup + 30.0,
            )
        )
    assert len(ledger.observations()) == 3
    assert len(ledger.observations(provider="modal")) == 2
    assert len(ledger.observations(provider="modal", gpu="H100")) == 2
    assert ledger.observations(provider="hpc-ai")[0].setup_s == pytest.approx(250.0)


def test_setup_fit_needs_three_samples(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path)
    for setup in (4.0, 6.0):
        ledger.record(
            Observation(provider="modal", t_submit=0.0, t_running=setup, t_ended=setup + 1)
        )
    assert SetupFit.from_ledger(ledger, provider="modal") is None


def test_setup_fit_reports_percentiles_once_three_samples_land(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path)
    for setup in (5.0, 8.0, 30.0):
        ledger.record(
            Observation(provider="modal", t_submit=0.0, t_running=setup, t_ended=setup + 1)
        )
    fit = SetupFit.from_ledger(ledger, provider="modal")
    assert fit is not None and fit.samples == 3
    assert fit.p50_s == pytest.approx(8.0)
    assert fit.p90_s > fit.p50_s
    assert fit.mean_s == pytest.approx(43 / 3)


def test_offers_query_by_hardware_across_providers() -> None:
    catalog = Catalog(
        (
            Offer(provider="modal", gpu="GB200", rate_usd_hr=11.0),
            Offer(
                provider="hpc-ai",
                gpu="GB200",
                region="sg",
                rate_usd_hr=7.5,
                granularity_s=60,
                minimum_s=60,
            ),
            Offer(provider="hpc-ai", gpu="H200", rate_usd_hr=2.2, granularity_s=60),
        )
    )
    gb200 = catalog.offers(gpu="gb200")
    assert {offer.provider for offer in gb200} == {"modal", "hpc-ai"}
    assert catalog.offers(gpu="GB200", provider="modal")[0].rate_usd_hr == pytest.approx(11.0)
    assert catalog.offers() == catalog.roster


def test_quotes_price_fitted_vs_unmeasured_providers(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path)
    for setup in (2.0, 2.0, 2.0):
        ledger.record(
            Observation(
                provider="modal", gpu="GB200", t_submit=0.0, t_running=setup, t_ended=setup + 1
            )
        )
    catalog = Catalog(
        (
            Offer(provider="modal", gpu="GB200", rate_usd_hr=11.0),
            Offer(provider="hpc-ai", gpu="GB200", rate_usd_hr=7.5, granularity_s=60, minimum_s=60),
        )
    )
    quotes = catalog.quotes(gpu="GB200", run_s=30.0, ledger=ledger, default_setup_s=300.0)
    assert (quotes[0].offer.provider, quotes[0].setup_samples) == ("modal", 3)
    assert quotes[0].expected_usd == pytest.approx(11.0 * 32 / 3600)
    assert quotes[1].offer.provider == "hpc-ai"
    assert quotes[1].expected_usd == pytest.approx(7.5 * 360 / 3600)
    assert quotes[1].p90_usd == pytest.approx(quotes[1].expected_usd)


def test_quotes_without_a_ledger_penalize_everyone_equally() -> None:
    catalog = Catalog((Offer(provider="modal", gpu="T4", rate_usd_hr=0.59),))
    quote = catalog.quotes(gpu="T4", run_s=10.0, default_setup_s=60.0)[0]
    assert quote.setup_samples == 0
    assert quote.expected_usd == pytest.approx(quote.p90_usd)


def test_catalog_add_extends_the_roster() -> None:
    catalog = Catalog()
    catalog.add(Offer(provider="modal", gpu="T4", rate_usd_hr=0.59))
    assert len(catalog.offers(gpu="T4")) == 1


def test_catalog_persists_and_reloads_offers(tmp_path: Path) -> None:
    catalog = Catalog(
        (
            Offer(
                provider="hpc-ai",
                gpu="B200-SXM-180GB",
                rate_usd_hr=3.5,
                granularity_s=60,
                source="scraped",
            ),
        )
    )
    target = tmp_path / "catalog.ndjson"
    catalog.save(target)
    reloaded = Catalog.load(target)
    assert reloaded.offers(gpu="B200-SXM-180GB")[0].rate_usd_hr == pytest.approx(3.5)
    assert Catalog.load(tmp_path / "missing.ndjson").roster == []
    Catalog().save(tmp_path / "empty.ndjson")
    assert Catalog.load(tmp_path / "empty.ndjson").roster == []


def test_gpuhunt_rows_import_as_offers() -> None:
    rows = [
        SimpleNamespace(
            provider="vastai", gpu_name="H100", gpu_count=4, price=7.6, spot=True, location="US"
        ),
        SimpleNamespace(
            provider="lambdalabs",
            gpu_name="H100",
            gpu_count=8,
            price=23.92,
            spot=False,
            location=None,
        ),
        SimpleNamespace(
            provider="broken", gpu_name="X", gpu_count=1, price=None, spot=False, location=""
        ),
    ]
    offers = from_gpuhunt(rows)
    assert len(offers) == 2
    assert offers[0].spot is True and offers[0].gpu_count == 4 and offers[0].region == "US"
    assert not offers[1].region
    assert offers[1].source == "imported:gpuhunt"


def test_gpuhunt_rows_import_under_the_name_the_probe_and_the_host_kind_use() -> None:
    """One provider name, so an imported row and a live probe answer the same catalog query."""
    rows = [
        SimpleNamespace(
            provider="vastai", gpu_name="H100", gpu_count=1, price=2.0, spot=False, location=""
        ),
        SimpleNamespace(
            provider="runpod", gpu_name="H100", gpu_count=1, price=2.5, spot=False, location=""
        ),
    ]
    probed = {"gpu_name": "H100", "num_gpus": 1, "dph_total": 1.8, "rentable": True}
    catalog = Catalog(tuple(from_gpuhunt(rows)) + tuple(from_vast([probed])))
    assert catalog_provider("vastai") == "vast"
    assert catalog_provider("runpod") == "runpod"
    assert {offer.source for offer in catalog.offers(provider="vast")} == {
        "imported:gpuhunt",
        "probed:vast",
    }
