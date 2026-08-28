import sys
from pathlib import Path

import pytest

from mainboard.trials import (
    Absent,
    Breach,
    Hunt,
    Miss,
    Optuna,
    Owed,
    Session,
    Study,
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
    """Both drivers stood in for, at the import call `driver` makes and nowhere wider.

    NOT `sys.modules`, and the reason is this suite itself: hypothesis is what most of these tests
    are written in, its pytest plugin imports the real module during every call phase, and a fake
    installed under that name takes the plugin down with it. Patching the one import call keeps
    the refusal path, the caching and the error handling of `driver` exactly as shipped.
    """
    stood = {"hypothesis": support_adaptive.hypothesis(), "optuna": support_adaptive.optuna()}
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


def test_a_study_writes_one_row_per_iteration_and_settles_on_its_worst_point(
    trial, session: Session, doubled: None
) -> None:
    """The sampler keeps no record, so every ask-tell iteration is a receipt row of its own."""
    proposer = Optuna({"tokens": [192, 288], "depth": [1024, 3072]}, seed=11)
    study = Study(
        trial,
        proposer,
        question="where does the served law miss worst",
        budget=4,
        seed=11,
        refuted="refuted",
        survived="undecided",
        owed=owed(),
        policy="default",
    )
    study.run(lambda tokens, depth: Miss(loss=tokens / depth, reading={"tokens": tokens}))

    written = rows(session)
    assert len(written) == 5
    assert [row["measured"]["row"] for row in written] == ["point"] * 4 + ["study"]
    assert [row["verdict"] for row in written] == ["undecided"] * 5
    assert proposer.study.told == [
        (0, 192 / 1024),
        (1, 288 / 3072),
        (2, 192 / 1024),
        (3, 288 / 3072),
    ]
    study_row = written[-1]["measured"]
    assert study_row["point"] == {"tokens": 192, "depth": 1024}
    assert study_row["outside"] is False and study_row["outside_points"] == 0
    assert study_row["owed"] is None and study_row["policy"] == "default"
    assert study_row["budget"] == 4 and study_row["seed"] == 11
    assert "found no excursion inside its budget" in written[-1]["reason"]


def test_a_worst_point_outside_the_band_settles_refuted_and_names_what_confirms_it(
    trial, session: Session, doubled: None
) -> None:
    """A sampler walks toward the corner it is rewarded for, so its worst point is a proposal."""
    study = Study(
        trial,
        Optuna({"tokens": [192, 288]}, seed=1),
        question="where does the served law miss worst",
        budget=3,
        seed=1,
        refuted="refuted",
        survived="undecided",
        owed=owed(),
    )
    point, miss = study.run(lambda tokens: Miss(loss=tokens, outside=tokens > 200))

    assert point == {"tokens": 288} and miss.outside
    settled = rows(session)[-1]
    assert settled["verdict"] == "refuted"
    assert settled["measured"]["outside_points"] == 1
    assert settled["measured"]["owed"]["cell"] == {"shape": "w96", "tokens": "288"}
    assert "OUTSIDE the band" in settled["reason"] and "a CANDIDATE, owed" in settled["reason"]


def test_a_study_settles_the_last_word_it_reached_and_not_the_first_row_it_narrated(
    trial, session: Session, doubled: None
) -> None:
    """A search narrates as it goes, so a terminal prints the word the study ended on."""
    Study(
        trial,
        Optuna({"tokens": [192, 288]}, seed=0),
        question="where does the served law miss worst",
        budget=2,
        seed=0,
        refuted="refuted",
        survived="undecided",
        owed=owed(),
    ).run(lambda tokens: Miss(loss=tokens, outside=tokens > 200))

    assert [word for _, word in trial.item.user_properties] == [
        "undecided",
        "undecided",
        "refuted",
    ]
    assert trial.settled == "refuted"
