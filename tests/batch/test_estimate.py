from typing import TYPE_CHECKING

import pytest

from mainboard.batch import BatchEstimate, Estimator, JobEstimate, TransferSet, platform
from mainboard.costs import Catalog, Ledger, Observation, Offer
from mainboard.dispatch import HostSetup
from mainboard.manifest import HostProfile
from mainboard.probe import HostFacts

from .conftest import declaring, spec

if TYPE_CHECKING:
    from pathlib import Path

    from mainboard import Board

# One rented offer and one owned target, the two halves of any real fleet.
_OFFER = Offer(provider="vast", gpu="RTX 4090", rate_usd_hr=0.36, granularity_s=1)


def priced(board: Board, ledger: Ledger, **job: object) -> JobEstimate:
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
