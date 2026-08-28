from collections.abc import Mapping
from pathlib import Path

import pytest

from mainboard.trials import Dataset, Declaration, findings
from mainboard.trials.lints import pinned

from .support import declaration


def settled(store: Dataset, run: str, *rows: Mapping[str, object]) -> None:
    """Write `rows` into `store` as one run's fragments, filling in what every receipt carries."""
    writer = store.writer(run, {"node": store.node})
    for row in rows:
        writer.write({"outcome": "passed", "verdict": "validated", "params": {}, **row})


@pytest.fixture
def claim(tmp_path: Path) -> Dataset:
    """One claim's store, its node named so a finding can say which claim it belongs to."""
    (tmp_path / "alpha").mkdir()
    return declaration(tmp_path).universe.dataset("alpha")


@pytest.fixture
def declared(tmp_path: Path) -> Declaration:
    """The workspace whose three settle words the lints read `refuted` out of."""
    return declaration(tmp_path)


def test_a_residue_pinned_to_one_ulp_of_a_constant_is_reported_as_an_identity(
    claim: Dataset, declared: Declaration
) -> None:
    """Does a lane whose quotient could only ever have come back one way get named?

    THE INSTANCE THAT EARNED IT is `fprev_recovery`'s two-routes lane, ruled GAP on 2026-08-29:
    it computes `p^T K p` two ways that are the same sum reassociated, reports a `relative_gap` of
    exactly 0.0 on one control and one ulp on the other, and settles `validated` on an event its
    own `Refutes if:` cannot produce. The sibling shape cost `crossing_cascade` a FATAL.
    """
    settled(
        claim,
        "run-1",
        {"lane": "test_two_routes", "key": "carried-fused", "measured": {"relative_gap": 0.0}},
        {"lane": "test_two_routes", "key": "chain", "measured": {"relative_gap": 5e-324}},
    )
    found = findings(claim, declared.words)
    identity = [one for one in found if one.lint == "identity"]
    assert len(identity) == 1
    assert identity[0].lane == "test_two_routes" and identity[0].node == "alpha"
    assert "`relative_gap`" in identity[0].detail and "settles `known`" in identity[0].detail


def test_a_band_that_is_the_range_of_its_own_scored_rows_is_reported_as_unfailable(
    claim: Dataset, declared: Declaration
) -> None:
    """Does an interval read off the rows it scores get named as a report and not a test?

    THE INSTANCE THAT EARNED IT is `accuracy_selection`'s exact band, ruled FATAL on 2026-08-29:
    `law_low` and `law_high` are the min and max of the thirty rows the lane scores, two of which
    sit on the edges and define them, so the registered refutation cannot fire on any input.
    """
    edges = {"law_low": 0.9995460306230508, "law_high": 1.003487963301741}
    settled(
        claim,
        "run-1",
        {
            "lane": "test_exact",
            "key": "k1",
            "measured": {"published": 0.9995460306230508, **edges},
        },
        {"lane": "test_exact", "key": "k2", "measured": {"published": 1.001, **edges}},
        {
            "lane": "test_exact",
            "key": "k16",
            "measured": {"published": 1.003487963301741, **edges},
        },
    )
    found = [one for one in findings(claim, declared.words) if one.lint == "unfailable"]
    assert len(found) == 1
    assert "law_high and law_low" in found[0].detail
    assert "`published`" in found[0].detail and "no row can leave it" in found[0].detail


def test_a_kill_that_never_fires_where_the_claim_dies_is_reported_as_uncovered(
    claim: Dataset, declared: Declaration
) -> None:
    """Does a lane carrying a kill get named when the claim's refutations sit off its grid?

    THE INSTANCE THAT EARNED IT is `carried_block_width` W1, ruled FATAL on 2026-08-29: its width
    lane runs five shapes that are all `M > 1`, and the four `M = 1` shapes where the sibling
    ladder lane recorded the pre-registration dying are the four the kill lane never visits, so
    the claim cannot be falsified at the only shapes where it is false.
    """
    settled(
        claim,
        "run-1",
        {"lane": "test_width", "key": "heuristic/m16/k256", "measured": {"carried": 9}},
        {"lane": "test_width", "key": "heuristic/m64/k256", "measured": {"carried": 9}},
        {"lane": "test_ladder", "key": "heuristic/m16/k256", "measured": {"plateau": 8}},
        {
            "lane": "test_ladder",
            "key": "heuristic/m1/k512",
            "verdict": "refuted",
            "measured": {"plateau": 2},
        },
    )
    found = [one for one in findings(claim, declared.words) if one.lint == "registered-kill"]
    assert len(found) == 1
    assert found[0].lane == "test_width" and "heuristic/m1/k512" in found[0].detail
    assert "its grid never contains" in found[0].detail


def test_a_lane_that_moved_its_readings_and_can_die_is_left_alone(
    claim: Dataset, declared: Declaration
) -> None:
    """Does a healthy claim come back clean, so a finding means something when one appears?

    Both lanes span the same grid, both measured a quantity that moved, and each settled a
    refutation somewhere, which is every one of the three shapes absent at once.
    """
    settled(
        claim,
        "run-1",
        {"lane": "test_a", "key": "one", "measured": {"ratio": 1.2}},
        {"lane": "test_a", "key": "two", "verdict": "refuted", "measured": {"ratio": 3.4}},
        {"lane": "test_b", "key": "one", "measured": {"ratio": 0.5}},
        {"lane": "test_b", "key": "two", "verdict": "refuted", "measured": {"ratio": 9.1}},
    )
    assert findings(claim, declared.words) == ()


def test_a_store_that_took_no_reading_and_a_single_row_answer_nothing(
    claim: Dataset, declared: Declaration
) -> None:
    """Is one row a constant by arithmetic rather than a finding, and an empty store silent?

    A lane with one receipt cannot have moved anything, and reporting it would bury the real
    findings under one per lane of every fresh claim.
    """
    assert findings(claim, declared.words) == ()
    settled(
        claim,
        "run-1",
        {"lane": "test_one", "key": "only", "measured": {"gap": 0.0}},
        {"lane": "test_two", "key": "broken", "outcome": "failed", "measured": {}},
    )
    assert findings(claim, declared.words) == ()


@pytest.mark.parametrize(
    ("values", "constant"),
    [
        ((), None),
        ((1.0,), None),
        ((1.0, 1.0), 1.0),
        ((0.0, 0.0, 0.0), 0.0),
        ((1.0, 1.0 + 2**-52), 1.0),
        ((1.0, 1.5), None),
        ((0.0, 5e-324), 0.0),
    ],
    ids=[
        "nothing",
        "one_reading",
        "two_equal",
        "a_zero_that_never_moved",
        "one_ulp_apart",
        "genuinely_different",
        "the_smallest_subnormal_off_zero",
    ],
)
def test_a_constant_is_read_to_one_ulp_because_a_cancelling_product_arrives_rounded(
    values: tuple[float, ...], constant: float | None
) -> None:
    """Does the pin tolerate the last bit, which is where an algebraic cancellation lands?

    Demanding exactness would miss every real telescoping product, since the terms cancel in
    algebra and still travel through floating point.
    """
    assert pinned(values) == constant


def test_a_payload_that_is_not_a_number_is_never_read_as_one(
    claim: Dataset, declared: Declaration
) -> None:
    """Are booleans, strings and nulls left out of a constancy read rather than coerced?

    `True` is `1` in python and a lane's `same_tree: true` on every row is a fact about the
    claim rather than a residue, so counting it would report every honest agreement gate. An
    infinity is not a constant anything converged on, and a payload that is not an object at all
    carries no keys to read.
    """
    settled(
        claim,
        "run-1",
        {
            "lane": "test_gate",
            "key": "a",
            "measured": {"same_tree": True, "why": "ok", "n": None, "blew_up": float("inf")},
        },
        {
            "lane": "test_gate",
            "key": "b",
            "measured": {"same_tree": True, "why": "ok", "n": None, "blew_up": float("inf")},
        },
        {"lane": "test_gate", "key": "c", "measured": ["not", "an", "object"]},
    )
    assert findings(claim, declared.words) == ()
