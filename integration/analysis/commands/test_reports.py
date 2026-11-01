"""The analysis commands print through `Output`: text for a reader, JSON under `--json`."""

import json
from pathlib import Path

import pytest

from mainboard.analysis.commands.graphs.uses import uses
from mainboard.analysis.commands.tools import sql
from mainboard.render import Output

_QUERY = (
    "SELECT 1 AS one, NULL AS none, DATE '2020-01-01' AS day, [1, 2] AS pair, E'a\\tb' AS cell"
)


@pytest.fixture
def project(tmp_path: Path) -> Path:
    (tmp_path / "pyproject.toml").write_text("[tool.mcmr]\n", encoding="utf-8")
    (tmp_path / "module.py").write_text("class Name:\n    pass\n", encoding="utf-8")
    return tmp_path


@pytest.mark.parametrize(
    ("output", "printed"),
    [
        pytest.param(
            Output(),
            'one\tnone\tday\tpair\tcell\n1\t\t2020-01-01\t[1, 2]\t"a\\tb"\n',
            id="a-header-and-a-row-with-its-empty-cell-kept",
        ),
        pytest.param(
            Output(json_=True),
            '[{"one":1,"none":null,"day":"2020-01-01","pair":[1,2],"cell":"a\\tb"}]\n',
            id="a-json-array-of-rows",
        ),
    ],
)
def test_sql_prints_every_selected_column_as_rows_or_json(
    project: Path, output: Output, printed: str, capsys: pytest.CaptureFixture[str]
) -> None:
    sql(_QUERY, project, output=output)

    assert capsys.readouterr().out == printed


@pytest.mark.parametrize(
    ("output", "printed"),
    [
        pytest.param(Output(), "one\n", id="the-header-alone"),
        pytest.param(Output(json_=True), "[]\n", id="an-empty-json-array"),
    ],
)
def test_sql_answers_a_query_without_rows_by_its_columns(
    project: Path, output: Output, printed: str, capsys: pytest.CaptureFixture[str]
) -> None:
    sql("SELECT 1 AS one WHERE false", project, output=output)

    assert capsys.readouterr().out == printed


def test_a_report_prints_one_document_as_json_whose_text_stays_for_a_reader(
    project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    uses("Name", root=project, output=Output())
    text = capsys.readouterr().out
    uses("Name", root=project, output=Output(json_=True))
    document = json.loads(capsys.readouterr().out)

    assert text.startswith("module.Name  class  module.py:1")
    assert [found["qualname"] for found in document["definitions"]] == ["Name"]
