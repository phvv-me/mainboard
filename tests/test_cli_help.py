import sys
from typing import TYPE_CHECKING

import pytest

from mainboard import MissionError
from mainboard.cli import build
from mainboard.help import Help

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        ([], "One interface for environments"),
        (["query"], "--out"),
        (["batch", "run"], "--only"),
        (["ARTIFACTS"], "query"),
        (["--", "--max-usd"], "submit"),
    ],
)
def test_help_reads_live_commands_without_a_workspace(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    arguments: list[str],
    expected: str,
) -> None:
    with pytest.raises(SystemExit, match="^0$"):
        build(tmp_path)(["help", *arguments])
    assert expected in capsys.readouterr().out


def test_help_reports_no_match(tmp_path: Path) -> None:
    with pytest.raises(MissionError, match="no help matches"):
        build(tmp_path)(["help", "unfindable-search-word"])


@pytest.mark.parametrize(
    ("query", "location", "excerpt"),
    [
        ("Sensor.reading", "danger.py:5", "Read quartz observations"),
        ("zephyr", "README.md:3", "zephyr observations"),
    ],
)
def test_help_reads_shipped_files_without_importing_api_modules(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    query: str,
    location: str,
    excerpt: str,
) -> None:
    (tmp_path / "README.md").write_text("# Observations\n\nRead zephyr observations.\n")
    (tmp_path / "danger.py").write_text(
        'raise RuntimeError("this module must not be imported")\n\n'
        "class Sensor:\n"
        "    @property\n"
        "    def reading(self):\n"
        '        """Read quartz observations without constructing the sensor."""\n'
        '        raise RuntimeError("nor may a descriptor execute")\n'
    )
    discovery = Help(build(tmp_path))
    discovery.package = tmp_path
    discovery.show(query)
    output = capsys.readouterr().out
    assert f"{tmp_path / location}" in output
    assert excerpt in output
    assert not (tmp_path / "__pycache__").exists()


@pytest.mark.parametrize(
    "body",
    ["zephyr\n\nquartz\n", "```python\nzephyr()\nquartz()\n```\n", "| zephyr |\n| quartz |\n"],
)
def test_help_respects_document_boundaries_and_exact_command_precedence(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], body: str
) -> None:
    (tmp_path / "README.md").write_text(f"# Guide\n\n{body}")
    discovery = Help(build(tmp_path))
    discovery.package = tmp_path
    with pytest.raises(MissionError, match="no help matches"):
        discovery.show("zephyr quartz")
    (tmp_path / "invalid.py").write_text("not valid Python source !!!\n")
    discovery.show("batch run")
    assert "--only" in capsys.readouterr().out


def test_missing_plot_dependencies_name_the_local_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delitem(sys.modules, "mainboard.plotting", raising=False)
    monkeypatch.setitem(sys.modules, "paleta", None)
    with pytest.raises(MissionError, match=r"mainboard\[wandb,plot\].*packages/paleta"):
        build(tmp_path)(
            ["plot", "SELECT 1 AS x, 2 AS y", "--x", "x", "--y", "y", "--out", "plot.png"]
        )
