"""The scorecard: dollars per useful GPU hour from the landings the lake kept."""

from collections.abc import Iterator
from math import inf, nan
from pathlib import Path

import pytest
from hypothesis import example, given
from hypothesis import strategies as st

from mainboard.costs import Ledger, Observation, Score, Scorecard, Track
from mainboard.state import Lake

from .builders import landings, offer

_COUNT = st.integers(min_value=0, max_value=30)
_SOME = st.integers(min_value=1, max_value=30)
_RATE = st.floats(min_value=0.05, max_value=40.0, allow_nan=False)


@pytest.fixture
def ledger(workspace: Path) -> Iterator[Ledger]:
    """A lake's ledger holding two landings of provider `p` and one that failed."""
    session = Lake.at(workspace).ready().session()
    kept = Ledger(session)
    for row in landings("p", landed=2, slow=1):
        kept.record(row)
    yield kept
    session.close()


def _leader(card: Scorecard, *kinds: str) -> str:
    """The provider among `kinds` whose equally priced offer ranks first."""
    scores = [card.score(offer(kind), kind=kind) for kind in kinds]
    return min(scores, key=lambda score: score.order).track.provider


def _held(delivered: float, *, setup_s: float, useful: float = 1.0) -> Score:
    """A score priced at `useful`, landing about `delivered` of the time in `setup_s`."""
    track = Track(provider="p", landings=100, landed=round(delivered * 100), ready_p50_s=setup_s)
    return Score(
        offer=offer("p"),
        track=track,
        useful_usd_hr=useful,
        job_usd=useful,
        billed_s=0,
        setup_s=setup_s,
    )


@given(
    rate=_RATE,
    count=st.integers(min_value=1, max_value=8),
    landed=_COUNT,
    slow=_COUNT,
    refused=_COUNT,
    ready_s=st.floats(min_value=1.0, max_value=3600.0),
)
@example(rate=2.0, count=1, landed=1, slow=1, refused=0, ready_s=inf)
@example(rate=2.0, count=1, landed=1, slow=1, refused=0, ready_s=nan)
def test_the_useful_price_never_falls_below_the_listed_price_per_gpu(
    rate: float, count: int, landed: int, slow: int, refused: int, ready_s: float
) -> None:
    rows = landings("p", landed=landed, slow=slow, refused=refused, ready_s=ready_s)
    score = Scorecard(rows).score(offer("p", rate_usd_hr=rate, gpu_count=count), kind="p")
    assert score.useful_usd_hr >= rate / count * (1 - 1e-12)


@given(landed=_COUNT, slow=_COUNT, rate=_RATE)
def test_a_better_landing_record_never_costs_more_per_useful_hour(
    landed: int, slow: int, rate: float
) -> None:
    surer = Scorecard(landings("p", landed=landed + 1, slow=slow))
    worse = Scorecard(landings("p", landed=landed, slow=slow + 1))
    cheap = surer.score(offer("p", rate_usd_hr=rate), kind="p")
    dear = worse.score(offer("p", rate_usd_hr=rate), kind="p")
    assert cheap.useful_usd_hr <= dear.useful_usd_hr


@given(landed=_COUNT, slow=_COUNT)
def test_equally_priced_offers_rank_by_the_surer_landing_first(landed: int, slow: int) -> None:
    sure = landings("sure", landed=landed + 1, slow=slow)
    risky = landings("risky", landed=landed, slow=slow)
    assert _leader(Scorecard([*risky, *sure]), "risky", "sure") == "sure"


@given(landed=_SOME, slow=_COUNT)
def test_equally_reliable_offers_rank_by_the_sooner_ready_one_first(
    landed: int, slow: int
) -> None:
    quick = landings("quick", landed=landed, slow=slow, ready_s=30.0)
    late = landings("late", landed=landed, slow=slow, ready_s=90.0)
    assert _leader(Scorecard([*late, *quick]), "late", "quick") == "quick"


@pytest.mark.parametrize(
    ("first", "second"),
    [
        pytest.param(
            _held(0.9, setup_s=100.0), _held(0.8, setup_s=10.0), id="the-surer-landing-comes-first"
        ),
        pytest.param(
            _held(0.9, setup_s=10.0), _held(0.9, setup_s=100.0), id="the-sooner-ready-when-as-sure"
        ),
        pytest.param(
            _held(0.5, setup_s=99.0),
            _held(0.9, setup_s=10.0, useful=2.0),
            id="a-lower-cost-beats-both",
        ),
    ],
)
def test_offers_of_equal_cost_rank_by_reliability_then_by_ready_time(
    first: Score, second: Score
) -> None:
    assert first.order < second.order


def test_an_unmeasured_offer_ranks_behind_an_equally_priced_one_that_mostly_delivered() -> None:
    known = landings("known", landed=3, ready_s=300.0)
    assert _leader(Scorecard(known), "new", "known") == "known"


@pytest.mark.parametrize(
    ("minimum_s", "billed_s"),
    [
        pytest.param(3600, 3600, id="a-60-minute-minimum-bills-a-five-minute-job-an-hour"),
        pytest.param(1800, 1800, id="a-30-minute-minimum-bills-half-an-hour"),
        pytest.param(600, 720, id="a-minimum-below-setup-and-release-bills-the-wall-time"),
        pytest.param(0, 720, id="no-minimum-bills-setup-run-and-release"),
    ],
)
def test_a_billed_minimum_decides_what_a_five_minute_job_costs(
    minimum_s: int, billed_s: int
) -> None:
    held = offer("p", rate_usd_hr=3.0, minimum_s=minimum_s)
    score = Scorecard([]).score(held, kind="p", run_s=300)
    assert score.billed_s == billed_s
    assert score.job_usd == pytest.approx(3.0 * billed_s / 3600)


@given(rate=_RATE, landed=_SOME, failed=_SOME)
def test_a_refused_create_costs_less_than_a_failed_landing_that_billed(
    rate: float, landed: int, failed: int
) -> None:
    free = Scorecard(landings("p", landed=landed, refused=failed))
    billed = Scorecard(landings("p", landed=landed, slow=failed))
    cheap = free.score(offer("p", rate_usd_hr=rate), kind="p")
    dear = billed.score(offer("p", rate_usd_hr=rate), kind="p")
    assert cheap.useful_usd_hr < dear.useful_usd_hr


@pytest.mark.parametrize(
    ("row", "judged", "landed"),
    [
        pytest.param(Observation(provider="p", t_submit=0, verdict="landed"), 1, 1, id="landed"),
        pytest.param(
            Observation(provider="p", t_submit=0, verdict="wrong-card"), 1, 0, id="wrong-card"
        ),
        pytest.param(
            Observation(provider="p", t_submit=0, verdict="odd"), 1, 0, id="a-verdict-nobody-knows"
        ),
        pytest.param(
            Observation(provider="p", t_submit=0, t_running=5),
            1,
            1,
            id="a-job-that-ran-before-verdicts",
        ),
        pytest.param(
            Observation(provider="p", t_submit=0), 0, 0, id="a-job-that-never-ran-before-verdicts"
        ),
    ],
)
def test_a_row_counts_by_its_verdict_or_by_whether_its_job_started(
    row: Observation, judged: int, landed: int
) -> None:
    track = Scorecard([row]).track("p")
    assert (track.landings, track.landed) == (judged, landed)


def test_an_empty_ledger_scores_every_offer_at_the_prior_and_reports_nothing() -> None:
    card = Scorecard([])
    score = card.score(offer("p"), kind="p")
    assert (score.track.landings, score.track.delivered) == (0, 0.5)
    assert score.setup_s == 300.0 and score.useful_usd_hr > 0
    assert card.report("p") == {} and card.track("p").ready_p50_s is None


@pytest.mark.parametrize(
    ("asked", "on_card", "seen"),
    [
        pytest.param("RTX 4090", True, 2, id="the-card-spelled-as-it-was-recorded"),
        pytest.param("RTX_4090", True, 2, id="the-card-spelled-the-way-vast-searches"),
        pytest.param("H200", False, 5, id="a-card-never-landed-falls-back-to-the-whole-record"),
    ],
)
def test_a_track_is_the_cards_own_record_or_else_the_providers_whole_one(
    asked: str, on_card: bool, seen: int
) -> None:
    rows = [*landings("p", landed=2, gpu="RTX 4090"), *landings("p", landed=3, gpu="H100")]
    track = Scorecard(rows).track("p", gpu=asked)
    assert (track.gpu, track.landings) == (asked if on_card else "", seen)


def test_the_report_names_the_landing_columns_of_a_measured_provider() -> None:
    rows = [
        Observation(provider="p", t_submit=0, verdict="landed", ready_s=ready, probe_mbps=mbps)
        for ready, mbps in ((10.0, 100.0), (20.0, 200.0), (30.0, 900.0))
    ]
    report = Scorecard([*rows, *landings("p", refused=1)]).report("p")
    assert report == {
        "landings": 4,
        "landed": 3,
        "refused": 1,
        "ready_p50_s": 20.0,
        "ready_p90_s": pytest.approx(28.0),
        "mbps_p10": pytest.approx(120.0),
    }


def test_a_scorecard_reads_the_landings_a_lake_kept(ledger: Ledger) -> None:
    track = Scorecard.of(ledger).track("p", gpu="H100")
    assert (track.landings, track.landed, track.ready_p50_s) == (3, 2, 60.0)
