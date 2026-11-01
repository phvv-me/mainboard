"""A mirror takes a folder a plan renamed as the files under it, both ways: what it adopts from the
project and what it publishes into it. It carries the Ruff configuration its project reads from
outside, and links the places its checkers' settings name and nothing else beside the project, so
the tools run in it read what they read in place and a crowded folder costs it nothing."""

from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from mainboard.analysis.edits import Rename
from mainboard.analysis.engine.apply import Plan
from mainboard.analysis.engine.mirror import Mirror

from .builders import written

_TREE = ("pkg/a.py", "pkg/sub/b.py", "other.py")
# A project with a sibling, a folder of stubs and an unrelated folder beside it.
_BESIDE = {"project/module.py": "", "sibling/a.py": "", "stubs/s.pyi": "", "unrelated/u.py": ""}
# Settings that name no place a mirror links: one not there, an absolute one, and a table of
# another tool that names the stubs.
_UNNAMED = """\
[tool.pyrefly]
search-path = ["../none", "/usr"]

[tool.black]
include = ["../stubs"]
"""


def _files(root: Path) -> list[str]:
    """The files under `root`, its links to the trees beside a mirror left out."""
    found = (path for path in root.rglob("*") if path.is_file() and not path.is_symlink())
    return sorted(path.relative_to(root).as_posix() for path in found)


def _renamed(root: Path) -> Mirror:
    """A mirror of a project holding `_TREE`, with `pkg` renamed to `renamed` in it."""
    for name in _TREE:
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(f"# {name}\n", encoding="utf-8")
    mirror = Mirror(root)
    plan = Plan(mirror.path)
    plan.add([Rename(path="pkg", target="renamed")])
    mirror.apply(plan)
    return mirror


def test_a_folder_a_plan_renamed_is_published_by_its_files_leaving_no_empty_folder(
    tmp_path: Path,
) -> None:
    with closing(_renamed(tmp_path)) as mirror:
        mirror.publish(tmp_path, {"pkg", "renamed"})

    assert _files(tmp_path) == ["other.py", "renamed/a.py", "renamed/sub/b.py"]
    assert not (tmp_path / "pkg").exists()


def test_a_folder_a_plan_renamed_is_adopted_back_from_the_project_by_its_files(
    tmp_path: Path,
) -> None:
    with closing(_renamed(tmp_path)) as mirror:
        mirror.adopt(tmp_path, {"pkg", "renamed"})
        adopted = _files(mirror.path)

    assert adopted == sorted(_TREE)


@pytest.mark.parametrize(
    ("configurations", "carried"),
    [
        pytest.param(
            {
                "pyproject.toml": "[tool.ruff]\n",
                "project/ruff.toml": 'extend = "../pyproject.toml"\n',
            },
            ["pyproject.toml"],
            id="one-the-project-extends",
        ),
        pytest.param(
            {"ruff.toml": "", "project/pyproject.toml": "[project]\nname = 'p'\n"},
            ["ruff.toml"],
            id="the-nearest-above-a-project-with-none-of-its-own",
        ),
        pytest.param(
            {"ruff.toml": "", "project/pyproject.toml": "[tool.ruff]\nline-length = 120\n"},
            [],
            id="none-for-a-project-configuring-itself",
        ),
    ],
)
def test_a_mirror_carries_the_ruff_configuration_its_project_reads_from_outside(
    tmp_path: Path, configurations: dict[str, str], carried: list[str]
) -> None:
    for name, text in {**configurations, "project/module.py": ""}.items():
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text(text, encoding="utf-8")

    with closing(Mirror(tmp_path / "project")) as mirror:
        outside = [name for name in _files(mirror.top) if not name.startswith("project/")]

    assert outside == carried


@settings(max_examples=20)
@given(
    crowd=st.sets(
        st.text("abcdef012", min_size=1, max_size=5).map("beside-{}".format), max_size=40
    )
)
@example(crowd={f"beside-{number}" for number in range(300)})
def test_a_mirror_holds_only_its_project_however_crowded_the_folder_it_sits_in(
    crowd: set[str],
) -> None:
    with TemporaryDirectory() as scratch:
        root = written(Path(scratch), {"project/module.py": "", **dict.fromkeys(crowd, "")})
        with closing(Mirror(root / "project")) as mirror:
            held = sorted(entry.name for entry in mirror.top.iterdir())

    assert held == ["project"]


@pytest.mark.parametrize(
    ("configuration", "linked"),
    [
        pytest.param(
            {"project/pyproject.toml": '[tool.pyrefly]\nsearch-path = ["..", "../stubs"]\n'},
            ["sibling", "stubs", "unrelated"],
            id="a-search-root-above-the-project-takes-every-entry-in-it",
        ),
        pytest.param(
            {"project/ty.toml": '[environment]\nextra-paths = ["../stubs"]\n'},
            ["stubs"],
            id="a-ty-extra-path-takes-that-folder-alone",
        ),
        pytest.param(
            {"project/mypy.ini": "[mypy]\nmypy_path = src:../stubs\n"},
            ["stubs"],
            id="a-mypy-ini-path-list-takes-the-folder-outside-the-project",
        ),
        pytest.param(
            {"project/setup.cfg": "[mypy-pkg.*]\nmypy_path = ../sibling, ../stubs\n"},
            ["sibling", "stubs"],
            id="a-setup-cfg-mypy-section-takes-each-folder-it-lists",
        ),
        pytest.param(
            {"project/pyproject.toml": '[tool.mypy]\nmypy_path = ["../sibling"]\n'},
            ["sibling"],
            id="a-pyproject-mypy-table-takes-its-folder",
        ),
        pytest.param(
            {"project/ruff.toml": 'src = ["../sibling"]\n'},
            ["sibling"],
            id="a-ruff-source-root-takes-its-folder",
        ),
        pytest.param(
            {"project/pyproject.toml": _UNNAMED},
            [],
            id="a-missing-place-an-absolute-one-and-another-tools-table-take-nothing",
        ),
        pytest.param({}, [], id="no-settings-take-nothing"),
    ],
)
def test_a_mirror_links_the_places_the_checkers_settings_name_and_no_others(
    tmp_path: Path, configuration: dict[str, str], linked: list[str]
) -> None:
    written(tmp_path, {**_BESIDE, **configuration})

    with closing(Mirror(tmp_path / "project")) as mirror:
        beside = {
            entry.name: entry.is_symlink()
            for entry in mirror.top.iterdir()
            if entry.name != "project"
        }

    assert beside == dict.fromkeys(linked, True)


def test_a_mirror_of_a_mirror_keeps_the_places_its_projects_settings_name(tmp_path: Path) -> None:
    written(
        tmp_path, {**_BESIDE, "project/ty.toml": '[environment]\nextra-paths = ["../stubs"]\n'}
    )

    with closing(Mirror(tmp_path / "project")) as proven, closing(Mirror(proven.path)) as work:
        read = (work.top / "stubs" / "s.pyi").is_file()
        beside = sorted(entry.name for entry in work.top.iterdir() if entry.name != "project")

    assert (read, beside) == (True, ["stubs"])
