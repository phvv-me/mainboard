import json
from functools import partial
from pathlib import Path

import pytest

from mainboard.trials import (
    OPENED,
    Admissibility,
    Declaration,
    Outcome,
    Probed,
    Session,
    digested,
)
from mainboard.trials import session as session_module
from mainboard.trials.session import WORD, lane_of, params_of

from .support import PROBED, Item, Taken, declaration
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
    assert row["case_id"] == "test_holds[qwen]" and row["kind"] == "law"
    assert row["trial"] == "alpha/test_law.py::test_holds[qwen]"
    assert row["commit"] == PROBED["commit"] and row["tree"] == PROBED["tree"]
    assert row["source_digest"] == PROBED["source_digest"]
    assert row["baselines_digest"] == "baselines-of-alpha"
    assert row["admissibility"] == "admissible" and row["gate_digest"] == ""
    assert row["run"] == session.run and row["opened_at_ns"] == session.opened


def test_many_runs_opened_at_one_clock_value_are_still_distinct_and_still_time_ordered(
    declared: Declaration, probed: None
) -> None:
    """Does a run's identity survive a second in which many of them open?

    The identity was a second-resolution timestamp and eight hex characters, which is 32 bits of
    collision room under a name every reader also SORTED by. It is now a uuid7, so the identity
    is 128 bits, the name is still lexically time-ordered because a uuid7 opens with its own
    millisecond, and the ORDER is read off `opened`, which is a separate fact in nanoseconds.
    """
    opened = [Session(declared) for _ in range(64)]
    assert len({run.run for run in opened}) == len(opened)
    assert len({run.run[:16] for run in opened}) == 1
    assert [run.run for run in opened] == sorted(run.run for run in opened)
    assert [run.opened for run in opened] == sorted(run.opened for run in opened)
    assert all(run.common[OPENED] == run.opened for run in opened)


def test_a_trial_names_its_case_and_its_run_as_two_fields_that_mean_two_things(
    session: Session, tmp_path: Path
) -> None:
    """Could a reader join two receipts on the field that used to be called `run_id`?

    It held the last component of the pytest node id beside a `run` column already holding the
    actual run, so joining on it joined on the test case. Two trials of one run now agree on
    `run` and differ on `case_id`, and the full node id is in `trial` where it always was.
    """
    for name in ("one", "two"):
        session.trial(Item(f"alpha/t.py::{name}", tmp_path / "alpha" / "t.py")).validated("ok")
    rows = session.declared.universe.dataset("alpha").rows(session.run)
    assert {str(row["run"]) for row in rows} == {session.run}
    assert sorted(str(row["case_id"]) for row in rows) == ["one", "two"]
    assert sorted(str(row["trial"]) for row in rows) == ["alpha/t.py::one", "alpha/t.py::two"]


def test_the_registration_a_lane_gates_on_rides_on_the_receipt_it_decided(
    session: Session, tmp_path: Path
) -> None:
    """Can a reader tell a pre-registered gate from one edited into agreement afterwards?

    Only if the row that decided the verdict left a trace on the row that recorded it, so the
    registration is read THROUGH the trial and comes straight back, and the receipt carries its
    digest. A lane that gates on nothing carries the empty digest, which is the honest answer.
    """
    registered = {"label": "qwen", "law_low": 0.9, "law_high": 1.1}
    trial = session.trial(Item("alpha/t.py::one", tmp_path / "alpha" / "t.py"))
    assert trial.gate(registered) is registered
    trial.validated("inside the registered band", ratio=1.0)

    row = session.declared.universe.dataset("alpha").rows(session.run)[0]
    assert row["gate_digest"] == digested(registered)
    assert row["gate_digest"] != digested({**registered, "law_high": 1.2})
    assert row["baselines_digest"] == "baselines-of-alpha"


def test_a_session_on_a_tree_nobody_can_identify_says_so_and_writes_it_on_every_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Does a dirty run refuse, warn, or write rows that say what they are?

    It writes, and the rows say `dirty`, because refusing would make the tool useless for the
    work it is most used for and a warning in a scroll-back is not a filter. The heading says it
    too, since learning it from a coverage table three days later is learning it too late.
    """
    (tmp_path / "alpha").mkdir()
    monkeypatch.setattr(
        session_module, "Preflight", partial(Taken, admissibility=Admissibility.DIRTY)
    )
    session = Session(declaration(tmp_path))
    assert "INADMISSIBLE (dirty), these rows are scratch work" in session.heading

    session.trial(Item("alpha/t.py::one", tmp_path / "alpha" / "t.py")).validated("measured")
    store = session.declared.universe.dataset("alpha")
    assert store.rows(session.run)[0]["admissibility"] == "dirty"
    assert store.passing().is_empty()


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
