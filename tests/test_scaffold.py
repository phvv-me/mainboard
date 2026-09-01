import sys
from collections.abc import Sequence
from typing import TYPE_CHECKING

import pytest
from plumbum import local

from mainboard import Board, MissionError
from mainboard.engines.compile.backend import PixiEngine

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_subprocess import FakeProcess
    from pytest_subprocess.fake_popen import FakePopen

# What the fixture workspace declares: the first template, which is the default, renders under
# its own home, and the second declares none and lands at the workspace root. The third is on
# disk without being declared at all, the shape a one-off render names by path.
_FIRST = "templates/study"
_SECOND = "templates/tool"
_UNDECLARED = "elsewhere/oneoff"
_HOME = "studies"

_TASKS = "mainboard.tasks.toml"
_ROWS = 'sc-baseline = { run = "python -m experiments.baseline.run execute" }\n'


@pytest.fixture(autouse=True)
def pixi_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give Pixi-routed scaffold calls a real cross-platform executable for fake Popen."""
    monkeypatch.setattr(PixiEngine, "command", property(lambda self: local[sys.executable]))


@pytest.fixture
def templates(workspace: Path) -> None:
    """Every template on this disk, declared or not, enough of each to be recognised as one."""
    for declared in (_FIRST, _SECOND, _UNDECLARED):
        directory = workspace / declared
        directory.mkdir(parents=True)
        (directory / "copier.yml").write_text("_subdirectory: template\n")


def rendering(fp: FakeProcess, destination: Path, *, tasks: bool = True) -> None:
    """Stand in for copier, laying down what a real render leaves behind."""

    def render(process: FakePopen) -> None:
        destination.mkdir(parents=True)
        (destination / "README.md").write_text("rendered\n")
        if tasks:
            (destination / _TASKS).write_text(_ROWS)

    fp.register([fp.any()], callback=render)


def test_a_project_lands_under_its_templates_home_with_its_rows_read_back(
    workspace: Path, templates: None, fp: FakeProcess
) -> None:
    """The default template renders where it says and pastes nothing itself.

    The first declared template is the default, and the task rows it writes are read back
    for the caller to paste, since the root manifest is a hand-curated file and half of that
    edit landing on its own is worse than none of it.
    """
    rendering(fp, workspace / _HOME / "scratch-probe")
    made = Board(workspace).scaffold().render("Scratch Probe")
    assert made.project == "scratch-probe"
    assert made.path == str(workspace / _HOME / "scratch-probe")
    assert made.tasks == str(workspace / _HOME / "scratch-probe" / _TASKS)
    assert made.snippet == _ROWS
    assert made.paste == f"{workspace / 'mainboard.toml'} [tasks]"
    assert str(workspace / _FIRST) in " ".join(fp.calls[0])


@pytest.mark.parametrize(
    ("template", "source", "landing"),
    [
        ("", _FIRST, f"{_HOME}/probe"),
        ("tool", _SECOND, "probe"),
        (_FIRST, _FIRST, f"{_HOME}/probe"),
        (_UNDECLARED, _UNDECLARED, "probe"),
        ("gh:owner/templates.git", "gh:owner/templates.git", "probe"),
    ],
    ids=[
        "the workspace's first declared template is the default",
        "naming a template is naming its home too",
        "spelling out a declared template's own path is naming that template",
        "a path nobody declared is a template too",
        "a template to fetch reaches the engine exactly as written",
    ],
)
def test_where_a_template_resolves_from_decides_where_the_project_lands(
    workspace: Path, templates: None, fp: FakeProcess, template: str, source: str, landing: str
) -> None:
    """Resolving a remote template is the engine's job, so only a path on this disk is checked."""
    destination = workspace / landing
    rendering(fp, destination, tasks=False)
    made = Board(workspace).scaffold().render("probe", template=template)
    assert made.path == str(destination)
    expected_source = source if source.startswith("gh:") else str(workspace / source)
    assert expected_source in " ".join(fp.calls[0])


@pytest.mark.parametrize(
    ("given", "landing", "tasks", "present", "absent"),
    [
        (
            {},
            f"{_HOME}/scratch-probe",
            True,
            ("project_name=Scratch Probe", "description=Scratch Probe", "home=monorepo"),
            (),
        ),
        (
            {
                "dest": "apart",
                "description": "Measures one thing well.",
                "answers": {"home": "standalone", "first_paper": "d"},
            },
            "apart",
            False,
            (
                "description=Measures one thing well.",
                "home=standalone",
                "first_paper=d",
            ),
            ("home=monorepo",),
        ),
    ],
    ids=[
        "one argument stands in for the questionnaire",
        "the manifest settles what is always the same and a caller settles the rest",
    ],
)
def test_the_answers_come_from_the_name_the_workspace_and_the_caller(
    workspace: Path,
    templates: None,
    fp: FakeProcess,
    given: dict[str, str | dict[str, str]],
    landing: str,
    tasks: bool,
    present: Sequence[str],
    absent: Sequence[str],
) -> None:
    rendering(fp, workspace / landing, tasks=tasks)
    made = Board(workspace).scaffold().render("Scratch Probe", **given)
    staged = " ".join(fp.calls[0])
    assert "--defaults" in staged
    assert all(answer in staged for answer in present)
    assert not any(answer in staged for answer in absent)
    assert (made.snippet == _ROWS) is tasks
    assert (made.paste != "") is tasks


@pytest.mark.parametrize(
    ("declared", "occupied", "refusal"),
    [
        (True, True, "already exists"),
        (False, False, r"no project template at .*templates[\\/]study"),
        (False, False, "declares no templates"),
    ],
    ids=[
        "a generator that can overwrite a project is one nobody runs twice",
        "a missing template names the path, since that is usually a wrong root",
        "nothing to render is a fact about the manifest",
    ],
)
def test_a_render_that_cannot_happen_says_which_thing_is_in_the_way(
    workspace: Path, declared: bool, occupied: bool, refusal: str
) -> None:
    if declared:
        (workspace / _FIRST).mkdir(parents=True)
        (workspace / _FIRST / "copier.yml").write_text("_subdirectory: template\n")
    if occupied:
        (workspace / _HOME / "probe").mkdir(parents=True)
    if "declares no templates" in refusal:
        (workspace / "mainboard.toml").write_text('[workspace]\nname = "bare"\n')
    with pytest.raises(MissionError, match=refusal):
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
