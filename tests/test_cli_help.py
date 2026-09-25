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
        '''raise RuntimeError("this module must not be imported")

class Sensor:
    @property
    def reading(self):
        """Read quartz observations without constructing the sensor."""
        raise RuntimeError("nor may a descriptor execute")
'''
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


@pytest.mark.parametrize(
    ("absent", "raised", "match"),
    [
        ("matplotlib", MissionError, r"mainboard\[wandb,plot\]"),
        ("mainboard.plots.figure", ModuleNotFoundError, r"mainboard\.plots\.figure"),
    ],
    ids=["a plot dependency names the local install", "anything else surfaces as itself"],
)
def test_a_missing_plot_module_names_the_install_only_when_it_is_a_plot_dependency(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    absent: str,
    raised: type[Exception],
    match: str,
) -> None:
    """The install hint is the right answer to a missing extra and a misleading one to a broken
    install of this tool, which would send its reader to reinstall an extra they already have.

    The plotting modules are forgotten first, so the verb imports them afresh whatever an
    earlier test already loaded, and the missing module is the one their import really hits.
    """
    for name in ("mainboard.plots.figure", "mainboard.plots.panel", "mainboard.plots.table"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setitem(sys.modules, absent, None)
    with pytest.raises(raised, match=match):
        build(tmp_path)(
            ["plot", "SELECT 1 AS x, 2 AS y", "--x", "x", "--y", "y", "--out", "plot.png"]
        )


@pytest.mark.parametrize(
    ("arguments", "match"),
    [
        (["--figure", "demo", "SELECT 1"], "--figure cannot be combined"),
        (["SELECT 1 AS x", "--x", "x"], "requires --x and --y"),
        (["--figure", "ghost"], r"no figure 'ghost'; declared figures are \[\]"),
        (
            ["SELECT 1 AS x, 2 AS y", "--x", "x", "--y", "y", "--style", "ghost"],
            r"no plot style 'ghost'; declared styles are \[\]",
        ),
    ],
    ids=[
        "a figure with its own chart mappings",
        "a chart without both axes",
        "an undeclared figure",
        "an undeclared style",
    ],
)
def test_a_plot_request_that_cannot_mean_one_chart_is_refused_before_any_query_runs(
    tmp_path: Path, arguments: list[str], match: str
) -> None:
    """A figure owns its SQL and mappings, so mixing in a second source would silently lose one;
    and a misspelled figure or style names what is declared instead of falling back to a
    default the reader never asked for."""
    pytest.importorskip("mainboard.plots.figure", exc_type=ModuleNotFoundError)
    target = tmp_path / "refused.png"
    with pytest.raises(MissionError, match=match):
        build(tmp_path)(["plot", *arguments, "--out", str(target)])
    assert not target.exists()
