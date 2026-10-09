"""What a job declares beyond its code, and what never enters a commit.

A whole test file run as one job (`mb run path/test_x.py`) reads what its tests read, so it
declares their needs together: a test that reads data only the lake keeps gets it written back
whether it runs alone or with its file. The data the lake keeps is named by where projects write
it, so a source package that happens to be called `datasets` or `evidence` is still code.
"""

import ast
from pathlib import Path, PurePosixPath
from textwrap import dedent

import pytest

from mainboard.board import Board
from mainboard.jobs.declare import Declaration, declared
from mainboard.manifest.schema.git import GitPolicy

_MODULE = ast.parse(
    dedent(
        """
        from mainboard.jobs import job

        @job(needs=["data/a", "data/b"], fetch="out")
        def test_one(): ...

        @job(needs=["data/b", "data/c"], resources=["config"], fetch="out")
        def test_two(): ...

        def test_three(): ...
        """
    )
)


def test_a_test_file_declares_what_its_tests_declare_together() -> None:
    together = declared(_MODULE, "")

    assert together.needs == ("data/a", "data/b", "data/c")
    assert together.resources == ("config",)
    assert together.fetch == "out"
    assert declared(_MODULE, "test_three") == Declaration()


def test_tests_naming_different_fetch_paths_leave_the_file_without_one() -> None:
    module = ast.parse(
        dedent(
            """
            @job(fetch="one")
            def test_one(): ...

            @job(fetch="two")
            def test_two(): ...
            """
        )
    )

    assert declared(module, "").fetch == ""


def test_a_test_pulls_back_the_home_its_receipts_are_kept_in(workspace: Path) -> None:
    """A folder inside a test's home, declared or given, still brings its store home.

    Pulling the folder alone left the objects the receipts pin in the snapshot (cutok 631).
    """
    home = "lab/datasets/experiments/law"
    test = workspace / "lab" / "experiments" / "law" / "test_law.py"
    test.parent.mkdir(parents=True)
    test.write_text(f'@job(fetch="{home}/probe")\ndef test_law(): ...\n', encoding="utf-8")
    board = Board(workspace)
    command = "lab/experiments/law/test_law.py::test_law"

    assert board.results(None, command=command) == home
    assert board.results(f"{home}/other", command=command) == home
    assert board.results("data/elsewhere", command=command) == "data/elsewhere"
    assert board.results(f"{home}/probe", command="python -m law") == f"{home}/probe"


def _kept_out(path: str) -> bool:
    return any(PurePosixPath(path).full_match(pattern) for pattern in GitPolicy().never_commit)


@pytest.mark.parametrize(
    "path",
    [
        "datasets/experiments/node/run=1/part-0.parquet",
        "research/reproducibility/datasets/inputs/operands.bin",
        "research/lab/experiments/node/evidence/receipts.json",
        "src/anything/table.parquet",
    ],
)
def test_data_never_enters_a_commit(path: str) -> None:
    assert _kept_out(path)


@pytest.mark.parametrize(
    "path",
    [
        "src/mcmr/rules/general/contextual/performance/evidence/r1001.py",
        "scripts/datasets/github/github_dataset.py",
        "research/compression/evidence/expert_streaming_proof.py",
    ],
)
def test_a_source_package_named_like_data_is_still_code(path: str) -> None:
    assert not _kept_out(path)
