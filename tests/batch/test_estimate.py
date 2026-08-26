from typing import TYPE_CHECKING

import pytest

from mainboard import ExecutionPlan, MissionError
from mainboard.batch import BatchEstimate, Estimator, JobEstimate, TransferSet, platform
from mainboard.costs import Catalog, Ledger, Observation, Offer
from mainboard.dispatch import HostSetup
from mainboard.dispatch.backends import Market, ProviderBackend
from mainboard.dispatch.vocabulary import JobState, Resources
from mainboard.manifest import HostProfile
from mainboard.probe import HostFacts

from .support import declaring, spec

if TYPE_CHECKING:
    from pathlib import Path

    from mainboard import Board

# One rented offer and one owned target, the two halves of any real fleet.
_OFFER = Offer(provider="vast", gpu="RTX 4090", rate_usd_hr=0.36, granularity_s=1)


def priced(board: Board, ledger: Ledger, **job: str | int) -> JobEstimate:
    """One job's row, against a catalog holding the single declared offer."""
    estimator = Estimator(board, catalog=Catalog((_OFFER,)), ledger=ledger)
    declared = spec({"target": "gold", "command": "true", **job}).jobs[0]
    return estimator.row(declared, TransferSet(job=declared.name, target=declared.target))


def timed(ledger: Ledger, key: str, *setups: float) -> None:
    """Record one observation per `setups` entry, each a run that waited that long to start."""
    for at, setup in enumerate(setups):
        ledger.record(
            Observation(provider=key, t_submit=float(at), t_running=at + setup, t_ended=at + 900)
        )


@pytest.mark.parametrize(
    ("alias", "kind", "key"),
    [("vast", "vast", "vast"), ("rented", "vast", "vast"), ("gold", "ssh", "gold")],
    ids=[
        "a provider is fitted by its kind",
        "a provider alias still fits by kind",
        "a machine is fitted by its own alias",
    ],
)
def test_setup_times_are_fitted_per_platform_not_per_declaration(
    alias: str, kind: str, key: str
) -> None:
    """Every rental of a kind waits the same way; two ssh boxes wait nothing alike."""
    assert platform(alias=alias, kind=kind) == key


def test_a_target_nobody_has_timed_is_priced_pessimistically_and_says_so(
    lab: Board, tmp_path: Path
) -> None:
    """A number no observation supports would be worse than an assumption that admits itself."""
    row = priced(lab, Ledger(tmp_path), runtime_s=60)
    assert (row.setup_p50_s, row.setup_p90_s, row.setup_samples) == (300.0, 300.0, 0)
    assert (row.rate_usd_hr, row.expected_usd, row.p90_usd) == (0.0, 0.0, 0.0)
    assert (row.target, row.kind, row.runtime_s) == ("gold", "ssh", 60.0)


def test_three_recorded_dispatches_turn_the_assumption_into_a_fit(
    lab: Board, tmp_path: Path
) -> None:
    ledger = Ledger(tmp_path)
    timed(ledger, "gold", 4.0, 6.0, 20.0)
    row = priced(lab, ledger)
    assert (row.setup_p50_s, row.setup_samples) == (6.0, 3)
    assert row.setup_p90_s >= row.setup_p50_s


def test_a_rented_row_carries_the_meter_and_an_owned_one_carries_nothing(
    lab: Board, tmp_path: Path
) -> None:
    """Owned hardware is already paid for, so its row prices at zero instead of at a guess."""
    ledger = Ledger(tmp_path)
    declaring(lab, "vast", HostProfile(kind="vast"))
    estimator = Estimator(lab, catalog=Catalog((_OFFER,)), ledger=ledger)
    rental = spec({"target": "vast", "command": "true", "gpu_name": "RTX 4090", "runtime_s": 3600})
    row = estimator.row(rental.jobs[0], TransferSet(job="vast-1", target="vast"))
    assert row.rate_usd_hr == 0.36
    assert row.expected_usd == pytest.approx(0.36 * (3600 + 300) / 3600)
    assert row.p90_usd == row.expected_usd  # unfitted, so the tail is the same assumption
    assert row.hardware == "1x RTX 4090"


def test_a_job_asking_for_hardware_no_offer_covers_is_still_a_row(
    lab: Board, tmp_path: Path
) -> None:
    row = priced(lab, Ledger(tmp_path), gpu_name="GB200", gpus=2)
    assert (row.rate_usd_hr, row.expected_usd) == (0.0, 0.0)
    assert row.hardware == "2x GB200"


class Quoting(ProviderBackend, Market):
    """A registered provider whose market answers, so an empty roster can still be priced."""

    name = "quoting"
    asked: list[tuple[str, int]] = []
    offers: list[Offer] = []
    fault: Exception | None = None

    def cancel(self, handle: str) -> None: ...

    def catalog(self, *, gpu_name: str = "", gpus: int = 0, limit: int = 0) -> list[Offer]:
        Quoting.asked.append((gpu_name, gpus))
        if Quoting.fault is not None:
            raise Quoting.fault
        return list(Quoting.offers)

    def state(self, handle: str) -> JobState:
        return JobState(handle=handle, verdict="ok")

    def submit(self, plan: ExecutionPlan, command: str, resources: Resources) -> str:
        return "q-1"


def quoted(lab: Board, tmp_path: Path, *, catalog: Catalog) -> JobEstimate:
    """One rented job's row against `catalog`, on a host routed to the quoting backend."""
    declaring(lab, "rented", HostProfile(kind="quoting"))
    estimator = Estimator(lab, catalog=catalog, ledger=Ledger(tmp_path))
    rental = spec(
        {"target": "rented", "command": "true", "gpu_name": "A100", "gpus": 4, "runtime_s": 3600}
    )
    return estimator.row(rental.jobs[0], TransferSet(job="rented-1", target="rented"))


def test_an_empty_roster_is_priced_from_the_providers_own_market_rather_than_at_zero(
    lab: Board, tmp_path: Path
) -> None:
    """The defect that closed the paid lane: nothing ever wrote the roster, so every rate was $0.

    A zero rate is not a cheap job, it is an unpriced one, and the house rule that no paid
    dispatch happens before the estimate is read turned that into a silent block. The survey
    priced the same card live off the same key, so the estimate asks the same market.
    """
    Quoting.asked = []
    Quoting.fault = None
    Quoting.offers = [Offer(provider="quoting", gpu="A100", rate_usd_hr=1.20, available=True)]
    row = quoted(lab, tmp_path, catalog=Catalog())
    assert (row.rate_usd_hr, row.rate_source) == (1.20, "live")
    assert row.expected_usd == pytest.approx(1.20 * (3600 + 300) / 3600)
    # Narrowed to exactly what the job asked to rent, and kept, so the next estimate is free.
    assert Quoting.asked == [("A100", 4)]
    assert (lab.root / ".mainboard" / "catalog.ndjson").is_file()


def test_a_roster_that_already_prices_the_card_costs_no_round_trip_and_says_it_is_stored(
    lab: Board, tmp_path: Path
) -> None:
    """A stored price may name a machine somebody else has taken, so the row admits which it is."""
    Quoting.asked = []
    Quoting.fault = None
    stored = Catalog((Offer(provider="quoting", gpu="A100", rate_usd_hr=0.90),))
    row = quoted(lab, tmp_path, catalog=stored)
    assert (row.rate_usd_hr, row.rate_source) == (0.90, "catalog")
    assert Quoting.asked == []


@pytest.mark.parametrize(
    ("fault", "offers", "said"),
    [
        pytest.param(
            MissionError("set VAST_API_KEY"), [], "VAST_API_KEY", id="a-provider-unkeyed"
        ),
        pytest.param(OSError("no route to host"), [], "no route", id="a-provider-unreachable"),
        pytest.param(None, [], "no offer right now", id="a-market-with-nothing-matching"),
    ],
)
def test_a_price_nobody_could_get_says_why_instead_of_reading_as_free(
    lab: Board, tmp_path: Path, fault: Exception | None, offers: list[Offer], said: str
) -> None:
    """The row's zero has to mean unknown, since a reader budgets against this table."""
    Quoting.asked = []
    Quoting.fault = fault
    Quoting.offers = offers
    row = quoted(lab, tmp_path, catalog=Catalog())
    assert row.rate_usd_hr == 0.0
    assert row.rate_source.startswith("unpriced: ") and said in row.rate_source


class Marketless(ProviderBackend):
    """A provider that quotes no market, which is what hpc-ai and modal actually are."""

    name = "marketless"

    def cancel(self, handle: str) -> None: ...

    def state(self, handle: str) -> JobState:
        return JobState(handle=handle, verdict="ok")

    def submit(self, plan: ExecutionPlan, command: str, resources: Resources) -> str:
        return "m-1"


def test_a_provider_that_publishes_no_market_leaves_the_roster_alone_and_says_it_is_unpriced(
    lab: Board, tmp_path: Path
) -> None:
    """Neither hpc-ai nor modal quotes a market, so the row admits it rather than reading free."""
    declaring(lab, "silent", HostProfile(kind="marketless"))
    estimator = Estimator(lab, catalog=Catalog(), ledger=Ledger(tmp_path))
    rental = spec({"target": "silent", "command": "true", "gpu_name": "H100"})
    row = estimator.row(rental.jobs[0], TransferSet(job="silent-1", target="silent"))
    assert (row.rate_usd_hr, row.rate_source) == (0.0, "unpriced: no offer right now")
    assert estimator.catalog.roster == []


def test_owned_hardware_is_never_asked_for_a_price_it_does_not_have(
    lab: Board, tmp_path: Path
) -> None:
    """A machine already paid for prices at zero as a fact, so no market is contacted for it."""
    Quoting.asked = []
    assert priced(lab, Ledger(tmp_path)).rate_source == "owned"
    assert Quoting.asked == []


def test_the_hardware_column_prefers_what_onboarding_actually_found(
    lab: Board, tmp_path: Path
) -> None:
    """A ready host describes real hardware, so the row never has to guess from the request."""
    cache = lab.dispatcher.cache
    cache.save_host(HostSetup(host="gold", root="/repo"))
    assert priced(lab, Ledger(tmp_path), gpus=1, gpu_name="RTX 4090").hardware == "1x RTX 4090"
    cache.save_host(
        HostSetup(
            host="gold",
            root="/repo",
            hardware=HostFacts(
                schema_version=1, hostname="gold", memory_total_bytes=64_000_000_000
            ),
        )
    )
    assert priced(lab, Ledger(tmp_path)).hardware == "64 GB RAM"


def test_a_batch_adds_up_what_it_ships_and_what_it_will_cost(lab: Board, tmp_path: Path) -> None:
    declared = spec(
        {"target": "gold", "command": "a", "runtime_s": 60},
        {"target": "gold", "command": "b", "runtime_s": 60},
    )
    transfers = [
        TransferSet(job=job.name, target=job.target, wire_bytes=size)
        for job, size in zip(declared.jobs, (10, 32), strict=True)
    ]
    table = Estimator(lab, catalog=Catalog(), ledger=Ledger(tmp_path)).table(
        "smoke-1", declared.jobs, transfers
    )
    assert isinstance(table, BatchEstimate)
    assert (table.batch, table.wire_bytes) == ("smoke-1", 42)
    assert (table.expected_usd, table.p90_usd) == (0.0, 0.0)


def test_the_workspace_owns_the_catalog_and_the_ledger_an_estimate_reads(lab: Board) -> None:
    """Both files are the tool's own, so pricing a batch needs nothing passed in."""
    estimator = Estimator(lab)
    assert estimator.catalog.roster == []
    assert estimator.ledger.path == lab.root / ".mainboard" / "costs" / "costs.ndjson"
