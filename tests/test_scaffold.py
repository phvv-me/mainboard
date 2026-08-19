from typing import TYPE_CHECKING

import pytest

from mainboard import Board, MissionError

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_subprocess import FakeProcess

_TEMPLATE = "templates/research"
_TASKS = "mainboard.tasks.toml"
_ROWS = 'sc-baseline = { run = "python -m experiments.baseline.run execute" }\n'


@pytest.fixture
def template(workspace: Path) -> Path:
    """A project template where the verb looks for one, enough of it to be found."""
    directory = workspace / _TEMPLATE
    directory.mkdir(parents=True)
    (directory / "copier.yml").write_text("_subdirectory: template\n")
    return directory


def rendering(fp: FakeProcess, destination: Path, *, tasks: bool = True) -> None:
    """Stand in for copier, laying down what a real render leaves behind."""

    def render(process: object) -> None:
        destination.mkdir(parents=True)
        (destination / "README.md").write_text("rendered\n")
        if tasks:
            (destination / _TASKS).write_text(_ROWS)

    fp.register([fp.any()], callback=render)


def test_a_project_lands_beside_its_siblings_with_its_rows_read_back(
    workspace: Path, template: Path, fp: FakeProcess
) -> None:
    """The default home is `research/<slug>`, which is what lets one manifest own every task."""
    rendering(fp, workspace / "research" / "scratch-probe")
    made = Board(workspace).scaffold().render("Scratch Probe")
    assert made.project == "scratch-probe"
    assert made.path == str(workspace / "research" / "scratch-probe")
    assert made.snippet == _ROWS
    assert made.paste == f"{workspace / 'mainboard.toml'} [tasks]"


def test_the_answers_the_template_asks_for_come_from_the_name(
    workspace: Path, template: Path, fp: FakeProcess
) -> None:
    """One argument stands in for the questionnaire, which is the point of the verb."""
    rendering(fp, workspace / "research" / "scratch-probe")
    Board(workspace).scaffold().render("Scratch Probe")
    staged = " ".join(fp.calls[0])
    assert "project_name=Scratch Probe" in staged
    assert "description=Scratch Probe research project." in staged
    assert "home=monorepo" in staged
    assert "--defaults" in staged


def test_standalone_renders_the_home_that_carries_its_own_manifest(
    workspace: Path, template: Path, fp: FakeProcess
) -> None:
    """The flag chooses the template's answer, not a different template."""
    rendering(fp, workspace / "elsewhere", tasks=False)
    made = Board(workspace).scaffold().render("probe", standalone=True, dest="elsewhere")
    assert "home=standalone-mainboard" in " ".join(fp.calls[0])
    assert made.snippet == ""
    assert made.paste == ""


def test_a_description_given_is_the_description_carried(
    workspace: Path, template: Path, fp: FakeProcess
) -> None:
    """The one sentence a README and every task row repeat is worth saying once."""
    rendering(fp, workspace / "research" / "probe")
    Board(workspace).scaffold().render("probe", description="Measures one thing well.")
    assert "description=Measures one thing well." in " ".join(fp.calls[0])


def test_rendering_over_an_existing_directory_is_refused(workspace: Path, template: Path) -> None:
    """A generator that can overwrite a project is a generator nobody runs twice."""
    (workspace / "research" / "probe").mkdir(parents=True)
    with pytest.raises(MissionError, match="already exists"):
        Board(workspace).scaffold().render("probe")


def test_a_workspace_with_no_template_says_where_it_looked(workspace: Path) -> None:
    """The refusal names the path, since a missing template is usually a wrong root."""
    with pytest.raises(MissionError, match=f"no project template at .*{_TEMPLATE}"):
        Board(workspace).scaffold().render("probe")


def test_a_failing_render_reaches_the_caller_with_copier_own_output(
    workspace: Path,
    template: Path,
    fp: FakeProcess,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The engine's own complaint is the useful half, and the refusal says how to install it."""
    fp.register([fp.any()], returncode=127, stderr="copier: command not found\n")
    with pytest.raises(MissionError, match=r"`copier copy` failed.*mainboard install"):
        Board(workspace).scaffold().render("probe")
    assert "copier: command not found" in capsys.readouterr().err
