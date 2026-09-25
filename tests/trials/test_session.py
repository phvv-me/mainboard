import copy
import json
import os
import time
from functools import partial
from itertools import chain, repeat
from pathlib import Path

import pytest

from mainboard.trials import (
    OPENED,
    Admissibility,
    Busy,
    Declaration,
    Outcome,
    Probed,
    Session,
    digested,
)
from mainboard.trials import lease as lease_module
from mainboard.trials import session as session_module
from mainboard.trials.session import WORD, lane_of, params_of

from .support import PROBED, Item, Taken, cell, declaration
from .test_declaring import knob


def test_runtime_manifest_still_refuses_registration_changed_after_dispatch(
    research: Path,
) -> None:
    session = Session(declaration(research))
    (research / "alpha/node.md").write_text("changed after dispatch\n")
    with pytest.raises(RuntimeError, match="changed after Mainboard prepared"):
        session.manifest(research / "alpha/test_law.py")
    assert not session.manifests


def test_a_research_run_refuses_to_log_without_a_captured_source_bundle(
    tmp_path: Path, probed: None
) -> None:
    root = tmp_path / "experiments"
    (root / "alpha").mkdir(parents=True)
    session = Session(declaration(root, repo=tmp_path))
    with pytest.raises(RuntimeError, match="captured Mainboard source bundle"):
        session.manifest(root / "alpha/test_law.py")


def test_a_research_run_writes_one_manifest_per_claim_from_its_captured_source(
    research: Path, tmp_path: Path
) -> None:
    session = Session(declaration(research, repo=tmp_path))
    written = session.manifest(research / "alpha/test_law.py")
    assert written is not None
    assert session.manifest(research / "alpha/test_other.py") is written
    assert written.schema_name == "mainboard.run.v1"
    manifest = json.loads(written.read(tmp_path))
    assert manifest["run"] == session.run and manifest["opened_at_ns"] == session.opened
    assert manifest["registration"] == "experiments/alpha/node.md"
    assert [row["path"] for row in manifest["files"]] == ["experiments/alpha/node.md"]
    assert "closure" not in manifest["source"] and manifest["source"]["digest"] == "a" * 64
    assert manifest["hardware"]["id"] == "GPU-1111"
    assert manifest["environment"] == {"polars": "1.0"}


def test_a_run_derives_every_field_a_lane_would_otherwise_have_to_retype(
    session: Session, tmp_path: Path
) -> None:
    """A fact a test has to retype is a fact a test will eventually retype wrong."""
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
    """A uuid7 run id sorts by its leading millisecond; the order within one is `opened`, in ns."""
    opened = [Session(declared) for _ in range(64)]
    assert len({run.run for run in opened}) == len(opened)
    assert len({run.run[:16] for run in opened}) == 1
    assert [run.run for run in opened] == sorted(run.run for run in opened)
    assert [run.opened for run in opened] == sorted(run.opened for run in opened)
    assert all(run.common[OPENED] == run.opened for run in opened)


def test_a_trial_names_its_case_and_its_run_as_two_fields_that_mean_two_things(
    session: Session, tmp_path: Path
) -> None:
    """The former `run_id` held the test case, so joining on it joined on the case."""
    for name in ("one", "two"):
        session.trial(Item(f"alpha/t.py::{name}", tmp_path / "alpha" / "t.py")).validated("ok")
    rows = session.declared.universe.dataset("alpha").rows(session.run)
    assert {str(row["run"]) for row in rows} == {session.run}
    assert sorted(str(row["case_id"]) for row in rows) == ["one", "two"]
    assert sorted(str(row["trial"]) for row in rows) == ["alpha/t.py::one", "alpha/t.py::two"]


def test_the_registration_a_lane_gates_on_rides_on_the_receipt_it_decided(
    session: Session, tmp_path: Path
) -> None:
    """The receipt carries the gate's digest, telling a pre-registered gate from one edited
    into agreement afterwards."""
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
    """A dirty run writes rows that say `dirty` (refusing would make the tool useless for its
    main work, a warning is not a filter) and its heading says so too."""
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
    trial = session.trial(Item("t.py::one", tmp_path / "t.py"))
    with pytest.raises(AttributeError, match="not a declared settle word"):
        assert trial.ranked
    trial.settle("undecided", reason="the separation is below the noise floor", gap=0.001)
    assert trial.settled == "undecided"


def test_the_flag_column_says_which_question_it_answers(tmp_path: Path, probed: None) -> None:
    """`session_<flag>` describes the session and `<flag>` the reading: with only the former,
    24 of 40 rows of one claim carried a policy their reading was not taken under."""
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
    flag, state = knob("policy", "pinned")
    session = Session(declaration(tmp_path, flags=(flag,)))
    state["policy"] = "default"
    assert "first moved by a trial that settled no receipt" in session.close()


def _receipts(target: Path) -> list[dict[str, object]]:
    lines = target.read_text(encoding="utf-8").splitlines()
    return [json.loads(line)["trial_receipt"] for line in lines]


def test_a_clean_run_closes_quietly_and_preserves_immutable_parts(
    session: Session, tmp_path: Path
) -> None:
    store = session.declared.universe.dataset("alpha")
    for key in ("a", "b"):
        session.trial(Item(f"alpha/t.py::one[{key}]", tmp_path / "alpha" / "t.py")).validated("ok")
    assert len(store.parts) == 2
    assert session.close() == ""
    assert len(store.parts) == 2
    ledger = store.root / "latest.jsonl"
    assert all(row["run"] == session.run for row in _receipts(ledger))


def test_a_partial_run_never_replaces_the_ledger_and_still_lands_readably(
    declared: Declaration, probed: None, tmp_path: Path
) -> None:
    """A re-run of one lane must not make the ledger forget the lanes it did not touch."""
    store = declared.universe.dataset("alpha")
    ledger = store.root / "latest.jsonl"
    first = Session(declared)
    for name in ("one", "two"):
        first.trial(Item(f"alpha/t.py::{name}", tmp_path / "alpha" / "t.py")).validated("ok")
    first.close()
    assert all(row["run"] == first.run for row in _receipts(ledger))

    second = Session(declared)
    second.trial(Item("alpha/t.py::one", tmp_path / "alpha" / "t.py")).validated("again")
    second.close()

    assert all(row["run"] == first.run for row in _receipts(ledger))
    partial = store.root / f"partial-{second.run}.jsonl"
    assert [row["run"] for row in _receipts(partial)] == [second.run]
    assert {row["lane"] for row in store.rows(first.run)} == {
        "alpha/t.py::one",
        "alpha/t.py::two",
    }
    assert store.status("alpha/t.py::one", ("",), cell()).run == second.run


def test_a_claim_drops_what_it_held_the_moment_collection_leaves_it(
    session: Session, tmp_path: Path
) -> None:
    """The residency scope pytest could not give, taken off the file tree."""
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
    """A `complete` that did not name its machine would let a campaign skip three architectures."""
    assert session.heading == "evidence on Test Card (GPU-1111):"
    bare = Session(declaration(tmp_path))
    bare.common.update({"card": "", "card_name": "", "card_probed": str(Probed.ABSENT)})
    assert bare.heading == "evidence on no card:"
    assert bare.cell({}).probing == {"card": Probed.ABSENT, "model": Probed.UNASKED}
    assert bare.cell({"model": "qwen"}).values == {"card": "", "model": "qwen"}


def test_claim_takes_the_card_lease_and_close_releases_it(
    session: Session, tmp_path: Path
) -> None:
    lock = tmp_path / lease_module.filename()
    assert not lock.exists()
    session.claim()
    assert lock.read_text(encoding="utf-8").split()[0] == str(os.getpid())
    session.close()
    assert not lock.exists()


def test_claim_is_a_no_op_off_a_host_with_no_card(tmp_path: Path, probed: None) -> None:
    bare = Session(declaration(tmp_path))
    bare.common.update({"card": "", "card_name": "", "card_probed": str(Probed.ABSENT)})
    bare.claim()
    assert bare.leased is None
    assert not (tmp_path / lease_module.filename()).exists()


def test_claim_refuses_to_measure_beside_a_live_holder(session: Session, tmp_path: Path) -> None:
    lock = tmp_path / lease_module.filename()
    lock.write_text(f"{os.getpid()} {time.time()}", encoding="utf-8")
    with pytest.raises(Busy, match=str(os.getpid())):
        session.claim()


def test_the_two_readings_a_collected_item_is_split_into(tmp_path: Path) -> None:
    assert lane_of(Item("t.py::one[a-b]", tmp_path)) == ("t.py::one", "a-b")
    assert lane_of(Item("t.py::one", tmp_path)) == ("t.py::one", "")
    assert params_of(Item("t.py::one", tmp_path)) == {}
    assert params_of(Item("t.py::one[2]", tmp_path, {"n": 2})) == {"n": "2"}


def test_a_trial_that_settled_nothing_leaves_a_failed_row_rather_than_a_hole(
    session: Session, tmp_path: Path
) -> None:
    trial = session.trial(Item("alpha/t.py::one", tmp_path / "alpha" / "t.py"))
    trial.record("", reason="the instrument is what failed", measured={}, outcome=Outcome.FAILED)
    row = session.declared.universe.dataset("alpha").rows(session.run)[0]
    assert row["outcome"] == "failed" and row["verdict"] == ""
    assert json.loads(json.dumps(row["measured"])) == {}


def test_a_declaration_stamps_the_universe_root_unless_a_repository_is_named(
    tmp_path: Path,
) -> None:
    assert declaration(tmp_path).tree == tmp_path
    assert declaration(tmp_path, repo=tmp_path.parent).tree == tmp_path.parent


def test_a_claims_residue_at_close_is_returned_and_everything_below_it_still_runs(
    declared: Declaration, probed: None, tmp_path: Path
) -> None:
    """A residue once escaped `close` and stopped the ledger update. The card reads 0 until the
    claim drops at close, then 4096."""
    readings = chain([0, 0, 0], repeat(4096))
    restless = copy.copy(declared)
    object.__setattr__(restless, "resident", lambda: next(readings))
    run = Session(restless)
    store = restless.universe.dataset("alpha")
    for key in ("a", "b"):
        run.trial(Item(f"alpha/t.py::one[{key}]", tmp_path / "alpha" / "t.py")).validated("ok")
    refusal = run.close()
    assert "alpha did not release what it held: 4096 bytes" in refusal
    assert len(store.parts) == 2
    assert all(row["run"] == run.run for row in _receipts(store.root / "latest.jsonl"))
