import pytest

from mainboard.core.errors import MissionError
from mainboard.jobs import lanes
from mainboard.jobs.call import Fresh

CELLS = (
    lanes.Cell(
        nodeid="lane.py::test[0-gpt2-20]",
        key="0-gpt2-20",
        params={"repetition": "0", "model": "gpt2", "chars": "20"},
    ),  # fmt: skip
    lanes.Cell(
        nodeid="lane.py::test[0-gpt2-100]",
        key="0-gpt2-100",
        params={"repetition": "0", "model": "gpt2", "chars": "100"},
    ),  # fmt: skip
    lanes.Cell(
        nodeid="lane.py::test[0-qwen3-20]",
        key="0-qwen3-20",
        params={"repetition": "0", "model": "qwen3", "chars": "20"},
    ),  # fmt: skip
)


def test_grouping_by_a_parametrize_name_keeps_collection_order_within_each_value() -> None:
    groups = lanes.grouped(CELLS, by="model")
    assert [(group.name, group.ids) for group in groups] == [
        ("gpt2", ("0-gpt2-20", "0-gpt2-100")),
        ("qwen3", ("0-qwen3-20",)),
    ]


def test_slicing_without_a_name_fills_each_job_then_the_remainder() -> None:
    groups = lanes.grouped(CELLS, per_job=2)
    assert [(group.name, group.ids) for group in groups] == [
        ("slice0", ("0-gpt2-20", "0-gpt2-100")),
        ("slice1", ("0-qwen3-20",)),
    ]
    everything = lanes.Group(name="all", ids=tuple(cell.key for cell in CELLS))
    assert lanes.grouped(CELLS) == (everything,)


def test_an_unknown_parametrize_name_is_refused_by_the_cell_that_lacks_it() -> None:
    with pytest.raises(MissionError, match=r"0-gpt2-20\] has no parametrize value named 'corpus'"):
        lanes.grouped(CELLS, by="corpus")


def test_the_node_is_the_directory_right_under_experiments() -> None:
    target = "research/x/experiments/card_scaling/test_documents.py::test"
    assert lanes.node_of(target) == "card_scaling"
    assert lanes.node_of("packages/tool/tests/test_a.py::test") == ""


def test_collection_lines_round_trip_through_the_capture_and_the_parser() -> None:
    lines = [
        "collecting ...",
        *("CELL " + cell.model_dump_json() for cell in CELLS),
        "3 collected",
    ]
    text = "\n".join(lines) + "\n"
    assert lanes.parsed(text) == CELLS


def test_the_summary_lists_every_host_and_group_with_its_ids() -> None:
    plan = lanes.summary(["local", "gold"], lanes.grouped(CELLS, by="model"))
    assert [(row["host"], row["group"], row["cells"]) for row in plan] == [
        ("local", "gpt2", 2),
        ("local", "qwen3", 1),
        ("gold", "gpt2", 2),
        ("gold", "qwen3", 1),
    ]
    assert plan[0]["ids"] == "0-gpt2-20 0-gpt2-100"


def test_a_fresh_plan_reads_its_timeout_ids_and_pytest_arguments() -> None:
    plan = Fresh.parsed(["--fresh", "--timeout", "30", "0-gpt2", "1-gpt2", "--", "-q", "--rerun"])
    assert plan is not None
    assert (plan.ids, plan.args, plan.timeout) == (("0-gpt2", "1-gpt2"), ("-q", "--rerun"), 30.0)
    assert Fresh.parsed(["-q"]) is None
    with pytest.raises(SystemExit, match="parametrize ids"):
        Fresh.parsed(["--fresh", "--", "-q"])
