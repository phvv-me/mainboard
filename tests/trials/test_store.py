import json
from collections.abc import Mapping
from pathlib import Path

import pytest

from mainboard import Board
from mainboard.trials import (
    ADMISSIBILITY,
    OPENED,
    Ambiguous,
    Dataset,
    Declaration,
    Ledger,
    Universe,
    wire,
)
from mainboard.trials.dataset import Cell
from mainboard.trials.ledger import RECEIPTS_VAR
from mainboard.verdicts import TrialVerdict

from .support import cell


def taken(
    store: Dataset, run: str, *rows: Mapping[str, object], opened: int | None = None
) -> None:
    """Write `rows` into `store` as one run's fragments, filling in what every receipt carries.

    Admissible unless a row says otherwise, since that is what a session on a committed tree
    writes and the exclusion is the exception being tested rather than the default under test.

    opened: the creation coordinate this run persists, none when the run is only named.
    """
    common: dict[str, object] = {"node": store.node, ADMISSIBILITY: "admissible"}
    if opened is not None:
        common[OPENED] = opened
    writer = store.writer(run, common)
    for row in rows:
        writer.write({"outcome": "passed", "measured": {}, "params": {}, **row})


def test_a_wire_line_is_the_printed_contract_a_dispatch_boundary_reads() -> None:
    """One JSON object under one key, newline included, which is the whole of the contract."""
    line = wire({"run_id": "one", "outcome": "passed"})
    assert line.endswith("\n")
    assert json.loads(line) == {"trial_receipt": {"run_id": "one", "outcome": "passed"}}


def test_a_ledger_appends_its_receipts_and_frames_them_where_a_dispatch_staged_a_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The jsonl sink and the csv beside it, the split an evidence folder wants.

    The framing file is how a rented instance hands evidence back at all, since it keeps no
    workspace and returns a log rather than a directory.
    """
    framed = tmp_path / "framed.ndjson"
    monkeypatch.setenv(RECEIPTS_VAR, str(framed))
    ledger = Ledger(tmp_path / "raw", {"node": "alpha"})
    ledger.receipt({"run_id": "one", "outcome": "passed"})
    ledger.receipt({"run_id": "two", "outcome": "failed"})

    lines = (tmp_path / "raw" / "receipts.jsonl").read_text().splitlines()
    assert [json.loads(line)["trial_receipt"]["node"] for line in lines] == ["alpha", "alpha"]
    assert framed.read_text() == (tmp_path / "raw" / "receipts.jsonl").read_text()

    monkeypatch.delenv(RECEIPTS_VAR)
    Ledger(tmp_path / "raw", {"node": "beta"}).receipt({"run_id": "three"})
    assert len(framed.read_text().splitlines()) == 2
    assert len((tmp_path / "raw" / "receipts.jsonl").read_text().splitlines()) == 3

    ledger.table("rows.csv", [])
    assert not (tmp_path / "raw" / "rows.csv").exists()
    ledger.table("rows.csv", [{"n": 1, "seen": ["a", "b"]}])
    ledger.table("rows.csv", [{"n": 2, "seen": ["c"]}])
    assert (tmp_path / "raw" / "rows.csv").read_text().splitlines() == [
        "n,seen",
        '1,"[""a"", ""b""]"',
        '2,"[""c""]"',
    ]


def test_a_run_that_dies_keeps_every_trial_it_took_and_a_run_that_ends_pays_for_one_footer(
    store: Dataset, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One fragment per trial during a run, folded into one file once nothing can be lost.

    A one-row parquet file pays a whole footer and schema, so the fragments buy crash safety
    while the run is alive and cost many times the space after it. A second writer opened on the
    same partition counts what is there rather than overwriting the first writer's trials.
    """
    framed = tmp_path / "framed.ndjson"
    monkeypatch.setenv(RECEIPTS_VAR, str(framed))
    writer = store.writer("run-1", {"node": "alpha"})
    row = writer.write({"lane": "l", "key": "a", "outcome": "passed", "measured": {"n": 1}})
    assert row["run"] == "run-1" and row["measured"] == {"n": 1}
    writer.compact()
    assert len(writer.parts) == 1

    writer.write({"lane": "l", "key": "b", "outcome": "passed", "measured": {}})
    resumed = store.writer("run-1", {"node": "alpha"})
    assert resumed.written == 2
    resumed.write({"lane": "l", "key": "c", "outcome": "passed", "measured": {}})
    assert len(store.parts) == 3
    writer.compact()
    assert len(store.parts) == 1
    assert len(store.rows("run-1")) == 3
    assert len(framed.read_text().splitlines()) == 3


def test_a_store_reads_runs_written_before_a_column_existed_beside_runs_written_after(
    store: Dataset,
) -> None:
    """Two runs need not share a schema, and a missing axis is the same fact as an empty one.

    A campaign found this the hard way: one host held a run from before a provenance field
    existed and the whole store stopped collecting, because a plain scan refuses a union.
    """
    taken(store, "run-1", {"lane": "l", "key": "a"})
    taken(store, "run-2", {"lane": "l", "key": "a", "model": "qwen", "model_probed": "found"})
    assert store.runs == ("run-1", "run-2")
    assert store.newest == "run-2"

    frame = store.scan().collect()
    assert sorted(frame["model"]) == ["", "qwen"]
    assert sorted(frame["model_probed"]) == ["", "found"]
    assert len(store.rows()) == 1
    assert store.rows()[0]["model"] == "qwen"
    assert len(store.rows("run-1")) == 1


def test_an_empty_store_answers_with_nothing_rather_than_raising(store: Dataset) -> None:
    """Every reader tolerates a claim that has never written a receipt, since most have not."""
    assert not store.parts and store.runs == () and store.newest == ""
    assert store.rows() == [] and store.rows("run-1") == []
    assert not store.passing().columns
    assert store.status("l", ("a",), Cell()).state == "missing"
    assert store.lanes() == frozenset()
    target = store.root / "out.ndjson"
    target.parent.mkdir(parents=True, exist_ok=True)
    assert store.as_jsonl(target) == 0
    assert target.read_text() == ""


def test_full_asks_whether_one_run_alone_reproduces_every_lane_the_store_has_ever_known(
    store: Dataset,
) -> None:
    """The ledger guard's own question: would reminting from just this run drop a lane?"""
    taken(store, "run-1", {"lane": "a", "key": "x"}, {"lane": "b", "key": "x"})
    assert store.lanes() == frozenset({"a", "b"})
    assert store.lanes("run-1") == frozenset({"a", "b"})
    assert store.full("run-1") is True

    taken(store, "run-2", {"lane": "a", "key": "x"})
    assert store.lanes("run-2") == frozenset({"a"})
    assert store.lanes() == frozenset({"a", "b"})
    assert store.full("run-2") is False
    assert store.full("run-1") is True

    taken(store, "run-3", {"lane": "c", "key": "x"})
    assert store.full("run-1") is False


def test_a_store_is_found_from_its_partitions_or_from_the_evidence_directory_above_them(
    store: Dataset, tmp_path: Path
) -> None:
    """A person pointing a verb at their evidence should not have to know the layout."""
    taken(store, "run-1", {"lane": "l", "key": "a"})
    assert Dataset.holding(store.root) is not None
    found = Dataset.holding(store.root.parent, axes=("card",))
    assert found is not None and found.root == store.root and found.axes == ("card",)
    assert Dataset.holding(tmp_path) is None


def test_the_current_view_takes_the_newest_reading_of_each_cell_unless_every_sample_is_wanted(
    store: Dataset,
) -> None:
    """One row per cell is what a representative table draws, and averaging that is averaging one.

    A program whose cells owe several readings asks for all of them instead, which is the same
    store answering a different question rather than a second store.
    """
    where = cell(card="GPU-1", model="qwen").filters
    taken(store, "run-1", {"lane": "l", "key": "a", "measured": {"n": 1}, **where})
    taken(store, "run-2", {"lane": "l", "key": "a", "measured": {"n": 2}, **where})
    taken(store, "run-2", {"lane": "l", "key": "a", "outcome": "failed", "measured": {}, **where})

    current = store.passing()
    assert len(current) == 1
    assert json.loads(current["measured"][0]) == {"n": 2}
    assert len(store.passing(every=True)) == 2


def test_two_runs_inside_one_second_are_ordered_by_the_coordinate_and_not_by_the_suffix(
    store: Dataset,
) -> None:
    """Does recency follow what a run wrote down, or the random hex its name ends in?

    A run name opens with a second-resolution timestamp and ends in random hex, so ordering on
    the name orders two runs inside one second by their tail. Here the SECOND run to open sorts
    LEXICALLY FIRST, which is the case the old rule got backwards, and every recency question,
    the newest run, the one-row-per-cell view and the run a coverage read cites, has to follow
    the coordinate instead.
    """
    where = cell(card="GPU-1", model="qwen").filters
    second = "20260829T094024Z-aaaaaaaa"
    first = "20260829T094024Z-ffffffff"
    taken(store, first, {"lane": "l", "key": "a", "measured": {"n": 1}, **where}, opened=1_000)
    taken(store, second, {"lane": "l", "key": "a", "measured": {"n": 2}, **where}, opened=2_000)

    assert store.runs == (first, second)
    assert store.newest == second
    assert json.loads(store.passing()["measured"][0]) == {"n": 2}
    assert store.rows()[0]["measured"] == {"n": 2}
    assert store.status("l", ("a",), cell(card="GPU-1", model="qwen")).run == second


def test_a_store_that_cannot_order_two_runs_refuses_newest_rather_than_picking_one(
    store: Dataset,
) -> None:
    """Is an unresolvable tie an answer or a refusal?

    Two runs that opened at the same instant are not orderable, and a `newest` that picked one
    anyway would be picking by whatever the sort fell back on, which is the defect this whole
    coordinate exists to remove. An explicitly named run still reads, because the store holds
    both and the ambiguity is only about which one is current.
    """
    taken(store, "20260829T094024Z-aaaaaaaa", {"lane": "l", "key": "a"}, opened=7)
    taken(store, "20260829T094024Z-ffffffff", {"lane": "l", "key": "b"}, opened=7)
    with pytest.raises(Ambiguous, match="opened at the same instant"):
        assert store.newest
    with pytest.raises(Ambiguous, match="20260829T094024Z-aaaaaaaa"):
        store.passing()

    assert len(store.rows("20260829T094024Z-aaaaaaaa")) == 1
    assert store.stored == {"20260829T094024Z-aaaaaaaa", "20260829T094024Z-ffffffff"}
    assert store.retire("tied", ("20260829T094024Z-ffffffff",)).is_dir()
    assert store.newest == "20260829T094024Z-aaaaaaaa"


def test_a_run_written_before_the_coordinate_existed_is_dated_by_its_own_name(
    store: Dataset,
) -> None:
    """Can a store hold a generation from before recency was written down and still be read?

    Such a run knew its own second and never knew more, so that is what it answers with, and two
    of them one second apart order exactly as they always did. A run whose name encodes no
    instant at all is UNDATED: name order is the only statement it makes, it sorts before
    anything dated, and it can never tie with anything.
    """
    taken(store, "20260829T094024Z-old00000", {"lane": "l", "key": "a"})
    taken(store, "20260829T094025Z-old11111", {"lane": "l", "key": "b"})
    assert store.newest == "20260829T094025Z-old11111"

    taken(store, "handwritten", {"lane": "l", "key": "c"})
    assert store.runs[0] == "handwritten"
    assert store.newest == "20260829T094025Z-old11111"


def test_a_row_whose_tree_nobody_can_identify_is_visible_and_never_counts(
    store: Dataset,
) -> None:
    """Does scratch work stay readable while it stops being evidence?

    A run on a moving tree measures, prints and writes; what it must not do is satisfy a claim,
    because two dirty trees at one commit are one string and nothing tells them apart. A row from
    before this field existed reads `unrecorded`, which proves nothing either and so counts as
    nothing, rather than being quietly filled in as admissible.
    """
    where = cell(card="GPU-1", model="qwen")
    store.writer("run-dirty", {"node": store.node, ADMISSIBILITY: "dirty"}).write(
        {"lane": "l", "key": "a", "outcome": "passed", "measured": {}, **where.filters}
    )
    store.writer("run-old", {"node": store.node}).write(
        {"lane": "l", "key": "b", "outcome": "passed", "measured": {}, **where.filters}
    )
    assert len(store.rows("run-dirty")) == 1
    assert store.passing().is_empty()
    assert store.status("l", ("a", "b"), where).state == "missing"

    taken(store, "run-clean", {"lane": "l", "key": "a", **where.filters})
    assert len(store.passing()) == 1
    assert store.status("l", ("a",), where).state == "complete"


def test_coverage_is_asked_at_the_cell_and_a_second_card_never_satisfies_the_first(
    store: Dataset,
) -> None:
    """A key is a parametrize id and names no machine, which is what would publish rows twice.

    Two identical cards answer the same NAME, so the identity is the device uuid and a lane
    satisfied on one reads missing on the other.
    """
    here, there = cell(card="GPU-1", model="qwen"), cell(card="GPU-2", model="qwen")
    taken(store, "run-1", {"lane": "l", "key": "a", **here.filters})
    taken(store, "run-1", {"lane": "l", "key": "b", "outcome": "failed", **here.filters})

    mine = store.status("l", ("a", "b"), here)
    assert mine.state == "partial" and mine.have == 1 and mine.missing == ("b",)
    assert mine.run == "run-1" and mine.node == "alpha"
    assert store.status("l", ("a",), here).state == "complete"
    assert store.status("l", ("a",), there).state == "missing"
    assert store.status("other", ("a",), here).state == "missing"


def test_a_cell_that_owes_several_samples_accumulates_across_runs(tmp_path: Path) -> None:
    """One passing receipt completes a claim and does not complete a variance measurement.

    The target is declared per universe, the count is read across runs, and a partial cell is
    collected again so its new fragments join the ones already there rather than replacing them.
    """
    universe = Universe(root=tmp_path, axes=("card",), samples=3)
    store = universe.dataset("")
    where = cell(card="GPU-1")
    taken(store, "run-1", {"lane": "l", "key": "a", **where.filters})
    first = store.status("l", ("a",), where)
    assert (first.want, first.have, first.state) == (3, 1, "partial")

    taken(store, "run-2", {"lane": "l", "key": "a", **where.filters})
    taken(store, "run-3", {"lane": "l", "key": "a", **where.filters})
    settled = store.status("l", ("a",), where)
    assert (settled.want, settled.have, settled.state) == (3, 3, "complete")
    assert settled.run == "run-3"


def test_one_run_streams_out_as_the_json_lines_the_dispatch_boundary_already_reads(
    store: Dataset,
) -> None:
    """The adapter between the store at rest and the printed contract, one run at a time."""
    taken(store, "run-1", {"lane": "l", "key": "a", "measured": {"n": 1}})
    taken(store, "run-2", {"lane": "l", "key": "a", "measured": {"n": 2}})
    target = store.root / "out.ndjson"
    assert store.as_jsonl(target) == 1
    payload = json.loads(target.read_text())["trial_receipt"]
    assert payload["run"] == "run-2" and payload["measured"] == {"n": 2}
    assert store.as_jsonl(target, "run-1") == 1
    assert json.loads(target.read_text())["trial_receipt"]["measured"] == {"n": 1}


def test_a_retirement_takes_the_readable_ledger_with_it_and_remints_what_is_left(
    store: Dataset,
) -> None:
    """Does the one file in a store a person can read still describe the runs the store counts?

    A retirement that moved parquet fragments and left `latest.jsonl` behind hands that reader the
    exact generation the store stopped counting, which four universes were caught with on
    2026-08-29, so the ledger travels into the generation that owns it and is reminted from what
    survives.
    """
    taken(store, "run-1", {"lane": "l", "key": "a", "measured": {"n": 1}})
    taken(store, "run-2", {"lane": "l", "key": "a", "measured": {"n": 2}})
    ledger = store.root / "latest.jsonl"
    store.as_jsonl(ledger)
    assert json.loads(ledger.read_text())["trial_receipt"]["run"] == "run-2"

    generation = store.retire("old-lane-names", ("run-2",))
    assert generation == store.root.parent / "retired" / "generation=old-lane-names"
    assert (generation / "run=run-2").is_dir()
    assert json.loads((generation / "latest.jsonl").read_text())["trial_receipt"]["run"] == "run-2"
    assert store.runs == ("run-1",)
    assert json.loads(ledger.read_text())["trial_receipt"]["run"] == "run-1"


def test_retiring_a_run_a_store_never_held_names_it_rather_than_doing_nothing(
    store: Dataset,
) -> None:
    """Is a retirement of a run that is not here a typo the caller is told about?

    Silently succeeding would let a mistyped identity read as a completed retirement while the
    generation it was meant to remove stays in the current view.
    """
    taken(store, "run-1", {"lane": "l", "key": "a"})
    with pytest.raises(ValueError, match="holds no run run-9"):
        store.retire("nothing", ("run-9",))
    assert store.runs == ("run-1",)


def test_retiring_every_run_leaves_no_ledger_behind_to_describe_an_empty_store(
    store: Dataset,
) -> None:
    """Does emptying a store remove its ledger rather than leave one naming retired rows?"""
    taken(store, "run-1", {"lane": "l", "key": "a"})
    ledger = store.root / "latest.jsonl"
    store.as_jsonl(ledger)
    generation = store.retire("everything", ("run-1",))
    assert store.runs == () and not ledger.exists()
    assert (generation / "latest.jsonl").exists()


def test_a_store_that_never_kept_a_ledger_retires_without_inventing_one(store: Dataset) -> None:
    """Does a retirement work on a store whose receipts nobody ever streamed out?

    The ledger is a convenience a workspace opts into, so a store without one retires its runs
    and the survivors gain one rather than the retirement raising on a missing file.
    """
    taken(store, "run-1", {"lane": "l", "key": "a"})
    taken(store, "run-2", {"lane": "l", "key": "a"})
    generation = store.retire("first", ("run-1",))
    assert not (generation / "latest.jsonl").exists()
    assert json.loads((store.root / "latest.jsonl").read_text())["trial_receipt"]["run"] == "run-2"


def test_a_universe_finds_a_claim_off_the_file_tree_and_answers_flat_as_one_store(
    tmp_path: Path, declared: Declaration
) -> None:
    """The folder a lane sits in IS the claim it serves, so nothing is retyped anywhere."""
    universe = declared.universe
    assert universe.node_of(tmp_path / "alpha" / "test_law.py") == "alpha"
    assert universe.node_of(tmp_path / "test_flat.py") == ""
    assert universe.dataset("alpha").node == "alpha"
    assert universe.dataset("alpha").samples == 1
    assert universe.nodes == ()

    taken(universe.dataset("alpha"), "run-1", {"lane": "l", "key": "a"})
    assert universe.nodes == ("alpha",)

    flat = Universe(root=tmp_path / "flat")
    assert flat.nodes == ()
    taken(flat.dataset(""), "run-1", {"lane": "l", "key": "a"})
    assert flat.nodes == ("",)


def test_a_receipts_store_is_scored_one_run_at_a_time_rather_than_as_one_flat_stream(
    board: Board, tmp_path: Path
) -> None:
    """The verb catching up with the storage the harness already writes.

    A store holds every run a harness ever took, so reading them as one stream lets a lane that
    broke in one campaign condemn a clean re-run months later, with no flag able to dig it out.
    The newest run answers by default, `run` names an older one, and the heading says which.
    """
    store = Dataset(tmp_path / "evidence" / "receipts")
    store.writer("run-1", {"node": "alpha", "producer": "mainboard.trials"}).write(
        {"run_id": "one", "outcome": "failed", "verdict": "", "reason": "broke"}
    )
    store.writer("run-2", {"node": "alpha", "producer": "mainboard.trials"}).write(
        {"run_id": "one", "outcome": "passed", "verdict": "refuted", "reason": "it died"}
    )

    newest = board.verdicts().of(str(tmp_path / "evidence"))
    assert newest.stream.endswith("run run-2") and newest.code == 0
    assert newest.trials == (
        TrialVerdict(
            job="one",
            node="alpha",
            verdict="passed",
            settled="refuted",
            detail="it died",
            producer="mainboard.trials",
        ),
    )

    older = board.verdicts().of(str(tmp_path / "evidence" / "receipts"), run="run-1")
    assert older.code == 1 and older.trials[0].verdict == "failed"

    absent = board.verdicts().of(str(tmp_path / "evidence"), run="run-9")
    assert absent.code == 3 and "holds no receipts for run 'run-9'" in absent.note
