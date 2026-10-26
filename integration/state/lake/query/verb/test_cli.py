import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest
from dirty_equals import IsJson, IsPartialDict, IsStr
from PIL import Image

from mainboard import MissionError, Results

from ..footer import Footer
from ..rendering import Rendering

_RANGE = "SELECT * FROM range(20)"
_NULLS = "SELECT NULL AS absent, '' AS empty, 'NULL' AS literal"
_SUMMED = "SELECT sum(i) AS n FROM range(1000000000000) t(i)"
_PLOTTED = "SELECT i AS x, i * 2 AS y FROM range(5) t(i)"
_UNCAPPED = ["--max-rows", "0", "--max-bytes", "0"]


@pytest.mark.parametrize(
    ("sql", "flags", "options"),
    [
        pytest.param(_RANGE, [], {}, id="default-caps"),
        pytest.param(
            _RANGE,
            ["--max-rows", "6", "--max-bytes", "0"],
            {"max_rows": 6, "max_bytes": 0},
            id="row-cap",
        ),
        pytest.param(
            _RANGE,
            ["--max-rows", "-1", "--max-bytes", "-1"],
            {"max_rows": -1, "max_bytes": -1},
            id="caps-lifted-by-minus-one",
        ),
        pytest.param(
            _RANGE,
            ["--json", "--max-rows", "6", "--max-bytes", "0"],
            {"mode": "json", "max_rows": 6, "max_bytes": 0},
            id="json-row-cap",
        ),
        pytest.param(_NULLS, [], {}, id="null-empty-and-literal-null"),
        pytest.param(_NULLS, ["--human"], {"mode": "human"}, id="human"),
        pytest.param(
            "SELECT * FROM range(33)",
            ["--human", *_UNCAPPED],
            {"mode": "human", "max_rows": 0, "max_bytes": 0},
            id="human-streaming-every-batch",
        ),
    ],
)
def test_the_cli_prints_what_the_renderer_renders(
    mb, render: Callable[..., Rendering], sql: str, flags: list[str], options: dict[str, str | int]
) -> None:
    rendered = render(sql, **options)
    ran = mb("query", sql, *flags)
    assert ran.code == 0, ran.said
    assert (ran.out, ran.err) == (rendered.out, rendered.err)


@pytest.mark.parametrize(
    ("flags", "printed"),
    [
        pytest.param(
            [],
            IsStr(
                regex=r"n\n# at least 0 rows \(query timed out\); sha256=[0-9a-f]{64} "
                r"\(partial\); snapshot_id=0 .*\n"
            ),
            id="tsv-prints-its-header-and-a-partial-footer",
        ),
        pytest.param(
            ["--json", *_UNCAPPED],
            IsJson([{"_truncated": IsPartialDict(timed_out=True, complete=False)}]),
            id="json-finishes-a-parseable-partial-array",
        ),
        pytest.param(["--out", "timed-out.csv"], "", id="export-publishes-no-incomplete-file"),
    ],
)
def test_a_timed_out_query_fails_after_printing_a_labelled_partial_result(
    mb, workspace: Path, flags: list[str], printed: str
) -> None:
    ran = mb("query", _SUMMED, "--timeout", "0.02", *flags, timeout=15)
    assert ran.code != 0 and "timed out" in ran.err, ran.said
    assert ran.out == printed
    assert list(workspace.iterdir()) == [workspace / "mb.toml"]


@pytest.mark.parametrize(
    "flags",
    [
        pytest.param([], id="tsv"),
        pytest.param(["--json"], id="json"),
        pytest.param(["--human"], id="human"),
        pytest.param(["--out", "duplicates.json"], id="export"),
    ],
)
def test_duplicate_column_names_are_rejected(mb, workspace: Path, flags: list[str]) -> None:
    sql = "SELECT 1 AS id, 2 AS id"
    ran = mb("query", sql, *flags)
    assert ran.code != 0 and "unique AS alias" in ran.err and not ran.out
    assert not (workspace / "duplicates.json").exists()
    with pytest.raises(MissionError, match="duplicate column names"):
        Results(workspace).stamped(sql)


@pytest.mark.parametrize(
    "option",
    [
        pytest.param(["--json"], id="json"),
        pytest.param(["--human"], id="human"),
        pytest.param(["--max-rows", "1"], id="max-rows"),
        pytest.param(["--max-bytes", "1"], id="max-bytes"),
    ],
)
def test_export_rejects_every_display_option(mb, workspace: Path, option: list[str]) -> None:
    ran = mb("query", "SELECT 1", "--out", "out.csv", *option)
    assert ran.code != 0 and "--out exports all rows" in ran.err
    assert not (workspace / "out.csv").exists()


def test_multiple_statements_keep_the_programmatic_contract_and_cli_error(mb, workspace) -> None:
    with pytest.raises(ValueError, match="one statement"):
        Results(workspace).query("SELECT 1; SELECT 2")
    ran = mb("query", "SELECT 1; SELECT 2")
    assert ran.code == 1 and "one statement" in ran.err
    assert not (workspace / ".mb").exists()


def test_query_help_documents_snapshot_and_remote_timeout_scope(mb) -> None:
    help_text = " ".join(mb("query", "--help").out.split())
    assert all(
        text in help_text
        for text in (
            "not snapshot-pinned",
            "server limit",
            "attached read-only",
            "session scratch tables",
            "AT (VERSION => n)",
            "recovers snapshots",
        )
    )


def test_a_consumer_closing_the_pipe_stops_query_without_a_traceback(workspace: Path) -> None:
    command = [
        sys.executable,
        "-c",
        "from mainboard.cli import main; main()",
        "query",
        "SELECT * FROM range(1000000000000)",
        *_UNCAPPED,
        "--timeout",
        "0",
    ]
    with subprocess.Popen(
        command, cwd=workspace, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    ) as child:
        assert child.stdout is not None and child.stderr is not None
        assert child.stdout.readline() == "range\n"
        child.stdout.close()
        assert child.wait(timeout=15) == 0
        assert not child.stderr.read()


def test_plot_records_the_same_result_identity(mb, workspace: Path) -> None:
    expected = Footer.of(mb("query", _PLOTTED).out)
    target = workspace / "figure.png"
    plotted = mb("paper", "plot", _PLOTTED, "--x", "x", "--y", "y", "--out", str(target))
    assert plotted.code == 0 and Footer.of(plotted.out) == expected, plotted.said
    metadata = json.loads(target.with_name("figure.png.metadata.json").read_text(encoding="utf-8"))
    assert metadata["results"] == [IsPartialDict(**expected.model_dump(), complete=True)]
    with Image.open(target) as image:
        assert json.loads(image.info["Description"])["results"] == metadata["results"]
    assert Results(workspace).query(_PLOTTED).fetchall() == [(i, i * 2) for i in range(5)]
