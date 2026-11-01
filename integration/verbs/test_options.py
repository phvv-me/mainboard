"""A verb that reads SQL takes a positional statement or `--file`, never both."""

import json
from pathlib import Path

import pytest

_PLOT = ("data", "paper", "plot", "--x", "x", "--y", "y")


@pytest.mark.parametrize(
    "verb",
    [pytest.param(("data", "query"), id="query"), pytest.param(_PLOT, id="plot")],
)
def test_a_verb_given_a_statement_and_a_file_refuses_before_it_reads_either(
    mb, workspace: Path, verb: tuple[str, ...]
) -> None:
    before = sorted(workspace.iterdir())
    ran = mb(*verb, "SELECT 1 AS one", "--file", "missing.sql")
    assert ran.code == 1
    assert "SQL statement and --file are mutually exclusive" in ran.err
    assert (ran.out, sorted(workspace.iterdir())) == ("", before)


def test_a_query_reads_its_statement_from_a_file(mb, workspace: Path) -> None:
    (workspace / "one.sql").write_text("SELECT 1 AS one", encoding="utf-8")
    ran = mb("data", "query", "--file", "one.sql", "--json")
    assert (ran.code, json.loads(ran.out)) == (0, [{"one": 1}])
