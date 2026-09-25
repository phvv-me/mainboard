import pytest

from mainboard.trials import Dataset, Declaration, findings
from mainboard.trials.lints import pinned

from .support import taken


def test_a_residue_pinned_to_one_ulp_of_a_constant_is_reported_as_an_identity(
    store: Dataset, declared: Declaration
) -> None:
    """The instance: `fprev_recovery`'s two-routes lane, ruled GAP on 2026-08-29.

    It computes `p^T K p` two ways that are one sum reassociated, a `relative_gap` of exactly 0.0
    on one control and one ulp on the other.
    """
    taken(
        store,
        "run-1",
        {"lane": "test_two_routes", "key": "carried-fused", "measured": {"relative_gap": 0.0}},
        {"lane": "test_two_routes", "key": "chain", "measured": {"relative_gap": 5e-324}},
    )
    identity = [one for one in findings(store, declared.words) if one.lint == "identity"]
    assert len(identity) == 1
    assert identity[0].lane == "test_two_routes" and identity[0].node == "alpha"
    assert "`relative_gap`" in identity[0].detail and "settles `known`" in identity[0].detail


def test_a_band_that_is_the_range_of_its_own_scored_rows_is_reported_as_unfailable(
    store: Dataset, declared: Declaration
) -> None:
    low, high = 0.9995460306230508, 1.003487963301741
    edges = {"law_low": low, "law_high": high}
    taken(
        store,
        "run-1",
        {"lane": "test_exact", "key": "k1", "measured": {"published": low, **edges}},
        {"lane": "test_exact", "key": "k2", "measured": {"published": 1.001, **edges}},
        {"lane": "test_exact", "key": "k16", "measured": {"published": high, **edges}},
    )
    found = [one for one in findings(store, declared.words) if one.lint == "unfailable"]
    assert len(found) == 1
    assert "law_high and law_low" in found[0].detail
    assert "`published`" in found[0].detail and "no row can leave it" in found[0].detail


@pytest.mark.parametrize(
    ("refutation_key", "expected"),
    [("heuristic/m1/k512", 1), ("heuristic/m16/k256", 0), ("", 0), (None, 0)],
    ids=["named-off-grid", "named-in-grid", "unparameterized", "missing-key"],
)
def test_a_kill_that_never_fires_where_the_claim_dies_is_reported_as_uncovered(
    store: Dataset, declared: Declaration, refutation_key: str | None, expected: int
) -> None:
    """An absent key names no shape, and a refutation inside the grid is already covered."""
    taken(
        store,
        "run-1",
        {"lane": "test_width", "key": "heuristic/m16/k256", "measured": {"carried": 9}},
        {"lane": "test_width", "key": "heuristic/m64/k256", "measured": {"carried": 9}},
        {"lane": "test_ladder", "key": "heuristic/m16/k256", "measured": {"plateau": 8}},
        {
            "lane": "test_ladder",
            "key": refutation_key,
            "verdict": "refuted",
            "measured": {"plateau": 2},
        },
    )
    found = [one for one in findings(store, declared.words) if one.lint == "registered-kill"]
    assert len(found) == expected
    if expected:
        assert refutation_key is not None
        assert found[0].lane == "test_width" and refutation_key in found[0].detail
        assert "its grid never contains" in found[0].detail


def test_a_lane_that_moved_its_readings_and_can_die_is_left_alone(
    store: Dataset, declared: Declaration
) -> None:
    taken(
        store,
        "run-1",
        {"lane": "test_a", "key": "one", "measured": {"ratio": 1.2}},
        {"lane": "test_a", "key": "two", "verdict": "refuted", "measured": {"ratio": 3.4}},
        {"lane": "test_b", "key": "one", "measured": {"ratio": 0.5}},
        {"lane": "test_b", "key": "two", "verdict": "refuted", "measured": {"ratio": 9.1}},
    )
    assert findings(store, declared.words) == ()


def test_a_store_that_took_no_reading_and_a_single_row_answer_nothing(
    store: Dataset, declared: Declaration
) -> None:
    """One row is a constant by arithmetic, and reporting it would bury the real findings."""
    assert findings(store, declared.words) == ()
    taken(
        store,
        "run-1",
        {"lane": "test_one", "key": "only", "measured": {"gap": 0.0}},
        {"lane": "test_two", "key": "broken", "outcome": "failed", "measured": {}},
    )
    assert findings(store, declared.words) == ()


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
    assert pinned(values) == constant


def test_a_payload_that_is_not_a_number_is_never_read_as_one(
    store: Dataset, declared: Declaration
) -> None:
    """`True` is `1` in python, and `same_tree: true` on every row is a fact, not a residue."""
    measured = {"same_tree": True, "why": "ok", "n": None, "blew_up": float("inf")}
    taken(
        store,
        "run-1",
        {"lane": "test_gate", "key": "a", "measured": measured},
        {"lane": "test_gate", "key": "b", "measured": measured},
        {"lane": "test_gate", "key": "c", "measured": ["not", "an", "object"]},
    )
    assert findings(store, declared.words) == ()
