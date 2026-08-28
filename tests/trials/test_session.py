import json
from pathlib import Path

import pytest

from mainboard.trials import Declaration, Outcome, Probed, Session
from mainboard.trials.session import WORD, lane_of, params_of

from .support import Item, declaration
from .test_declaring import knob


def test_a_run_derives_every_field_a_lane_would_otherwise_have_to_retype(
    session: Session, tmp_path: Path
) -> None:
    """A lane supplies measurements and the receipt carries the rest, because the rest is derived.

    Who ran, on what card, at which commit, under which trial of which claim is already known,
    and a fact a test has to retype is a fact a test will eventually retype wrong.
    """
    item = Item(
        "alpha/test_law.py::test_holds[qwen]",
        tmp_path / "alpha" / "test_law.py",
        {"model": "qwen"},
    )
    trial = session.trial(item)
    assert (trial.lane, trial.key) == ("alpha/test_law.py::test_holds", "qwen")
    trial.validated("the law held", ratio=1.5)

    assert trial.settled == "validated"
    assert item.user_properties == [(WORD, "validated")]
    row = session.declared.universe.dataset("alpha").rows(session.run)[0]
    assert row["node"] == "alpha" and row["producer"] == "mainboard.trials"
    assert row["card"] == "GPU-1111" and row["card_probed"] == "found"
    assert row["model"] == "qwen" and row["model_probed"] == "unasked"
    assert row["outcome"] == "passed" and row["verdict"] == "validated"
    assert row["measured"] == {"ratio": 1.5} and row["params"] == {"model": "qwen"}
    assert row["run_id"] == "test_holds[qwen]" and row["kind"] == "law"
    assert row["commit"] == "abc1234"


def test_an_undeclared_word_refuses_at_the_attribute_rather_than_writing_an_unreadable_row(
    session: Session, tmp_path: Path
) -> None:
    """The words are the consumer's, so the methods are, and a typo is caught where it is made."""
    trial = session.trial(Item("t.py::one", tmp_path / "t.py"))
    with pytest.raises(AttributeError, match="not a declared settle word"):
        assert trial.ranked
    trial.settle("undecided", reason="the separation is below the noise floor", gap=0.001)
    assert trial.settled == "undecided"


def test_the_flag_column_says_which_question_it_answers(tmp_path: Path, probed: None) -> None:
    """`session_<flag>` is a fact about the session and `<flag>` is a fact about the reading.

    A review of the reference found 24 of 40 rows of one claim carrying a policy their own
    reading was not taken under, because only the session-level value existed.
    """
    (tmp_path / "alpha").mkdir()
    flag, state = knob("policy", "pinned")
    session = Session(declaration(tmp_path, flags=(flag,)))
    assert session.baseline == {"policy": "pinned"}
    assert session.common["session_policy"] == "pinned"

    state["policy"] = "default"
    item = Item("alpha/t.py::one", tmp_path / "alpha" / "t.py")
    session.trial(item).refuted("it did not survive")
    row = session.declared.universe.dataset("alpha").rows(session.run)[0]
    assert row["session_policy"] == "pinned" and row["policy"] == "default"
    assert session.leaked == {"policy": "alpha/t.py::one"}

    refusal = session.close()
    assert "tracked flag(s) ended off baseline" in refusal
    assert "first moved by alpha/t.py::one" in refusal
    assert "mainboard.trials.held" in refusal


def test_a_flag_left_moved_by_nothing_that_settled_still_names_what_it_can(
    tmp_path: Path, probed: None
) -> None:
    """A session fixture can move a knob without any trial settling under it, and that is worse."""
    flag, state = knob("policy", "pinned")
    session = Session(declaration(tmp_path, flags=(flag,)))
    state["policy"] = "default"
    assert "first moved by a trial that settled no receipt" in session.close()


def test_a_clean_run_closes_quietly_and_compacts_what_it_wrote(
    session: Session, tmp_path: Path
) -> None:
    """Compaction runs unconditionally, since a run's fragments are worth what they are worth."""
    store = session.declared.universe.dataset("alpha")
    for key in ("a", "b"):
        session.trial(Item(f"alpha/t.py::one[{key}]", tmp_path / "alpha" / "t.py")).validated("ok")
    assert len(store.parts) == 2
    assert session.close() == ""
    assert len(store.parts) == 1


def test_a_claim_drops_what_it_held_the_moment_collection_leaves_it(
    session: Session, tmp_path: Path
) -> None:
    """The residency scope pytest could not give, taken off the file tree instead of node types.

    Every lane that measures asks for its evidence line, so no claim can start without the
    previous one's holdings having been dropped first.
    """
    (tmp_path / "beta").mkdir()
    session.trial(Item("alpha/t.py::one", tmp_path / "alpha" / "t.py"))
    assert session.staged.claim == "alpha"
    session.staged.kept("weights", lambda: "loaded")

    session.trial(Item("alpha/t.py::two", tmp_path / "alpha" / "t.py"))
    assert session.staged.held == {"weights": "loaded"}

    session.trial(Item("beta/t.py::three", tmp_path / "beta" / "t.py"))
    assert session.staged.claim == "beta" and not session.staged.held


def test_a_run_names_the_machine_it_is_scoped_to(
    session: Session, tmp_path: Path, probed: None
) -> None:
    """The heading names the machine every one of this run's readings is scoped to.

    A `complete` that did not say which machine satisfied it is the line that would let a
    campaign skip three architectures without measuring any of them.
    """
    assert session.heading == "evidence on Test Card (GPU-1111):"
    bare = Session(declaration(tmp_path))
    bare.common.update({"card": "", "card_name": "", "card_probed": str(Probed.ABSENT)})
    assert bare.heading == "evidence on no card:"
    assert bare.cell({}).probing == {"card": Probed.ABSENT, "model": Probed.UNASKED}
    assert bare.cell({"model": "qwen"}).values == {"card": "", "model": "qwen"}


def test_the_two_readings_a_collected_item_is_split_into(tmp_path: Path) -> None:
    """A lane and its key are the node id at the bracket, and a lane with no grid has no params."""
    assert lane_of(Item("t.py::one[a-b]", tmp_path)) == ("t.py::one", "a-b")
    assert lane_of(Item("t.py::one", tmp_path)) == ("t.py::one", "")
    assert params_of(Item("t.py::one", tmp_path)) == {}
    assert params_of(Item("t.py::one[2]", tmp_path, {"n": 2})) == {"n": "2"}


def test_a_trial_that_settled_nothing_leaves_a_failed_row_rather_than_a_hole(
    session: Session, tmp_path: Path
) -> None:
    """A broken instrument is the one thing that must not be silent, so it writes and it fails."""
    trial = session.trial(Item("alpha/t.py::one", tmp_path / "alpha" / "t.py"))
    trial.record("", reason="the instrument is what failed", measured={}, outcome=Outcome.FAILED)
    row = session.declared.universe.dataset("alpha").rows(session.run)[0]
    assert row["outcome"] == "failed" and row["verdict"] == ""
    assert json.loads(json.dumps(row["measured"])) == {}


def test_a_declaration_stamps_the_universe_root_unless_a_repository_is_named(
    tmp_path: Path,
) -> None:
    """A universe that is not itself a work tree names the tree its commit comes from."""
    assert declaration(tmp_path).tree == tmp_path
    named: Declaration = declaration(tmp_path, repo=tmp_path.parent)
    assert named.tree == tmp_path.parent
