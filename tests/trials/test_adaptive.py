import sys
from pathlib import Path

import pytest

from mainboard.trials import (
    Absent,
    Breach,
    Hunt,
    Owed,
    Session,
    adaptive,
    driver,
)

from . import support_adaptive
from .support import Item


@pytest.fixture
def trial(session: Session, tmp_path: Path):
    """One evidence line in the `alpha` claim, which every lane below settles through."""
    return session.trial(Item("alpha/test_law.py::test_hunts", tmp_path / "alpha" / "test_law.py"))


@pytest.fixture
def doubled(monkeypatch: pytest.MonkeyPatch) -> None:
    """The driver stood in for, at the import call `driver` makes and nowhere wider.

    NOT `sys.modules`, and the reason is this suite itself: hypothesis is what most of these tests
    are written in, its pytest plugin imports the real module during every call phase, and a fake
    installed under that name takes the plugin down with it. Patching the one import call keeps
    the refusal path and the error handling of `driver` exactly as shipped.
    """
    stood = {"hypothesis": support_adaptive.hypothesis()}
    monkeypatch.setattr(adaptive, "import_module", lambda name: stood[name])


def rows(session: Session) -> list[dict]:
    """This run's receipt rows for the `alpha` claim, in the order they were written."""
    return session.declared.universe.dataset("alpha").rows(session.run)


def owed() -> Owed:
    """The declared cell a candidate below owes its confirmation to."""
    return Owed(lane="alpha/test_law.py::test_confirms", cell={"shape": "w96"})


def test_a_missing_driver_refuses_by_naming_the_package_and_the_extra_that_ships_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bare ModuleNotFoundError three frames down does not tell a reader what to install."""
    monkeypatch.setitem(sys.modules, "optuna", None)
    with pytest.raises(Absent, match=r"pip install mainboard\[search\]"):
        driver("optuna", "search")


def test_a_present_driver_comes_back_as_the_module_itself(doubled: None) -> None:
    """The import path under test is the real one, and only what comes back through it is fake."""
    assert driver("hypothesis", "adversarial").HealthCheck == ("too_slow",)


def test_an_owed_confirmation_states_the_cell_it_names_and_says_so_when_it_names_none() -> None:
    """A candidate carries the debt in words, because the receipt is what a reader reaches for."""
    assert "alpha/test_law.py::test_confirms[shape=w96] on fresh seeds" in owed().stated
    bare = Owed(lane="alpha/test_law.py::test_confirms", seeds="two fresh")
    assert bare.stated.endswith("test_confirms on two fresh seeds")


def test_a_law_that_holds_settles_the_survival_word_with_the_draw_count_and_no_witness(
    trial, session: Session, doubled: None
) -> None:
    """Surviving a search is a statement about the search, so it is never a validation word."""
    hunt = Hunt(
        trial,
        law="the phase is independent of the delta",
        draws=12,
        seed=3,
        refuted="refuted",
        survived="undecided",
        owed=owed(),
        node="alpha",
    )
    assert hunt.against(lambda size: None, size=[1, 2, 4, 8]) is None

    row = rows(session)[0]
    assert row["verdict"] == "undecided" and row["outcome"] == "passed"
    assert row["measured"]["draws"] == 12 and row["measured"]["draws_taken"] == 12
    assert row["measured"]["seed"] == 3 and row["measured"]["node"] == "alpha"
    assert row["measured"]["witness"] == {} and row["measured"]["owed"] is None
    assert row["measured"]["lane_kind"] == "adversarial"
    assert "survived 12 of 12 budgeted edge-biased draws at seed 3" in row["reason"]
    assert "nothing here is coverage" in row["reason"]


def test_a_law_that_breaks_settles_refuted_with_the_shrunk_witness_and_the_cell_it_owes(
    trial, session: Session, doubled: None
) -> None:
    """A refutation that cannot be retyped is a rumour, so the minimal draw rides on the row."""

    def law(size: int) -> None:
        if size >= 8:
            raise Breach(f"the law fails at size {size}", size=size, margin=0.5)

    hunt = Hunt(
        trial,
        law="the phase is independent of the delta",
        draws=8,
        seed=2,
        refuted="refuted",
        survived="undecided",
        owed=owed(),
    )
    witness = hunt.against(law, size=[1, 2, 4, 8, 16])

    assert isinstance(witness, Breach) and witness.witness == {"size": 8, "margin": 0.5}
    row = rows(session)[0]
    assert row["verdict"] == "refuted"
    assert row["measured"]["witness"] == {"size": 8, "margin": 0.5}
    assert row["measured"]["broke_at"] == 2 and row["measured"]["shrinks"] > 0
    assert row["measured"]["owed"]["lane"] == "alpha/test_law.py::test_confirms"
    assert "BROKE at draw 2 of 8" in row["reason"] and "a CANDIDATE, owed" in row["reason"]


def test_a_hunt_replays_from_its_own_receipt_because_the_seed_is_what_walked_it(
    trial, session: Session, doubled: None
) -> None:
    """Two hunts at one seed draw one sequence, which is the whole reproducibility contract."""

    def walked() -> list[int]:
        drawn: list[int] = []
        Hunt(
            trial,
            law="the order is the seed's",
            draws=5,
            seed=7,
            refuted="refuted",
            survived="undecided",
            owed=owed(),
        ).against(lambda size: drawn.append(size), size=[1, 2, 4])
        return drawn

    assert walked() == walked()
    assert rows(session)[0]["measured"]["replay"] == "re-run this lane at seed 7 with draws=5"
