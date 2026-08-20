from typing import TYPE_CHECKING

import pytest

from mainboard import Board, MissionError

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_subprocess import FakeProcess

# What the fixture workspace declares: the first template, which is the default, renders under
# its own home, and the second declares none and lands at the workspace root.
_FIRST = "templates/study"
_SECOND = "templates/tool"
_HOME = "studies"

_TASKS = "mainboard.tasks.toml"
_ROWS = 'sc-baseline = { run = "python -m experiments.baseline.run execute" }\n'


@pytest.fixture
def templates(workspace: Path) -> None:
    """Both declared templates, enough of each to be recognised as one."""
    for declared in (_FIRST, _SECOND):
        directory = workspace / declared
        directory.mkdir(parents=True)
        (directory / "copier.yml").write_text("_subdirectory: template\n")


def rendering(fp: FakeProcess, destination: Path, *, tasks: bool = True) -> None:
    """Stand in for copier, laying down what a real render leaves behind."""

    def render(process: object) -> None:
        destination.mkdir(parents=True)
        (destination / "README.md").write_text("rendered\n")
        if tasks:
            (destination / _TASKS).write_text(_ROWS)

    fp.register([fp.any()], callback=render)


def test_a_project_lands_under_its_templates_home_with_its_rows_read_back(
    workspace: Path, templates: None, fp: FakeProcess
) -> None:
    """The first declared template is the default, and where it renders is its own to say."""
    rendering(fp, workspace / _HOME / "scratch-probe")
    made = Board(workspace).scaffold().render("Scratch Probe")
    assert made.project == "scratch-probe"
    assert made.path == str(workspace / _HOME / "scratch-probe")
    assert made.snippet == _ROWS
    assert made.paste == f"{workspace / 'mainboard.toml'} [tasks]"
    assert _FIRST in " ".join(fp.calls[0])


def test_the_answers_come_from_the_name_and_from_what_the_workspace_declared(
    workspace: Path, templates: None, fp: FakeProcess
) -> None:
    """One argument stands in for the questionnaire, which is the point of the verb."""
    rendering(fp, workspace / _HOME / "scratch-probe")
    Board(workspace).scaffold().render("Scratch Probe")
    staged = " ".join(fp.calls[0])
    assert "project_name=Scratch Probe" in staged
    assert "description=Scratch Probe" in staged
    assert "home=monorepo" in staged
    assert "--defaults" in staged


def test_an_answer_given_overrides_the_one_the_manifest_declares(
    workspace: Path, templates: None, fp: FakeProcess
) -> None:
    """The manifest settles what is always the same; a caller settles the rest."""
    rendering(fp, workspace / "elsewhere", tasks=False)
    made = (
        Board(workspace)
        .scaffold()
        .render("probe", dest="elsewhere", answers={"home": "standalone", "first_paper": "d"})
    )
    staged = " ".join(fp.calls[0])
    assert "home=standalone" in staged and "home=monorepo" not in staged
    assert "first_paper=d" in staged
    assert made.snippet == "" and made.paste == ""


def test_a_named_template_declaring_no_home_renders_at_the_workspace_root(
    workspace: Path, templates: None, fp: FakeProcess
) -> None:
    """Naming a template is naming everything about it, its home included."""
    rendering(fp, workspace / "probe", tasks=False)
    made = Board(workspace).scaffold().render("probe", template="tool")
    assert made.path == str(workspace / "probe")
    assert _SECOND in " ".join(fp.calls[0])


def test_a_declared_templates_own_path_still_gets_what_the_workspace_declared(
    workspace: Path, templates: None, fp: FakeProcess
) -> None:
    """Spelling out the directory is naming the same template, home and answers included."""
    rendering(fp, workspace / _HOME / "probe")
    made = Board(workspace).scaffold().render("probe", template=_FIRST)
    assert made.path == str(workspace / _HOME / "probe")
    assert "home=monorepo" in " ".join(fp.calls[0])


def test_a_template_nobody_declared_is_rendered_from_the_path_given(
    workspace: Path, fp: FakeProcess
) -> None:
    """A path is a template too, which is what keeps a one-off render one command."""
    oneoff = workspace / "elsewhere/oneoff"
    oneoff.mkdir(parents=True)
    (oneoff / "copier.yml").write_text("_subdirectory: template\n")
    rendering(fp, workspace / "probe", tasks=False)
    Board(workspace).scaffold().render("probe", template="elsewhere/oneoff")
    assert str(oneoff) in " ".join(fp.calls[0])


def test_a_template_to_fetch_reaches_the_engine_exactly_as_written(
    workspace: Path, fp: FakeProcess
) -> None:
    """Resolving a remote template is the engine's job, so nothing here checks a local path."""
    rendering(fp, workspace / "probe", tasks=False)
    Board(workspace).scaffold().render("probe", template="gh:owner/templates.git")
    assert "gh:owner/templates.git" in " ".join(fp.calls[0])


def test_a_description_given_is_the_description_carried(
    workspace: Path, templates: None, fp: FakeProcess
) -> None:
    """The one sentence a README and every task row repeat is worth saying once."""
    rendering(fp, workspace / _HOME / "probe")
    Board(workspace).scaffold().render("probe", description="Measures one thing well.")
    assert "description=Measures one thing well." in " ".join(fp.calls[0])


def test_rendering_over_an_existing_directory_is_refused(workspace: Path, templates: None) -> None:
    """A generator that can overwrite a project is a generator nobody runs twice."""
    (workspace / _HOME / "probe").mkdir(parents=True)
    with pytest.raises(MissionError, match="already exists"):
        Board(workspace).scaffold().render("probe")


def test_a_workspace_declaring_no_templates_says_so(workspace: Path) -> None:
    """Nothing to render is a fact about the manifest, so the refusal points back at it."""
    (workspace / "mainboard.toml").write_text('[workspace]\nname = "bare"\n')
    with pytest.raises(MissionError, match="declares no templates"):
        Board(workspace).scaffold().render("probe")


def test_a_declared_template_that_is_not_one_says_where_it_looked(workspace: Path) -> None:
    """The refusal names the path, since a missing template is usually a wrong root."""
    with pytest.raises(MissionError, match=f"no project template at .*{_FIRST}"):
        Board(workspace).scaffold().render("probe")


def test_a_failing_render_reaches_the_caller_with_copier_own_output(
    workspace: Path,
    templates: None,
    fp: FakeProcess,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The engine's own complaint is the useful half, and the refusal says how to install it."""
    fp.register([fp.any()], returncode=127, stderr="copier: command not found\n")
    with pytest.raises(MissionError, match=r"`copier copy` failed.*mainboard install"):
        Board(workspace).scaffold().render("probe")
    assert "copier: command not found" in capsys.readouterr().err
