"""The settings a project's checkers read: each of Ruff, ty, pyrefly and mypy takes the nearest
file that configures it (Ruff also the files that one extends), and a setting naming a place by a
relative path makes that place part of what a mirror must carry."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from mainboard.analysis.engine.settings import Settings

from .builders import written

_PLACES = ("alpha", "beta", "gamma")
# The project itself, an absolute path and a sibling that is not there name nothing outside.
_UNNAMED = """\
[environment]
extra-paths = ["src", ".", "/usr", "../none"]
"""


def _read(base: Path, files: dict[str, str]) -> Settings:
    """The settings of the project `ws/project` under `base`, which holds `files` beside it."""
    written(base, {"ws/project/module.py": "", **files})
    return Settings(base.resolve() / "ws" / "project")


@pytest.mark.parametrize(
    ("files", "outside"),
    [
        pytest.param({"ws/ty.toml": ""}, ["ws/ty.toml"], id="a-ty-file-beside-the-project"),
        pytest.param(
            {"ws/pyrefly.toml": "", "ws/project/pyproject.toml": "[tool.pyrefly]\n"},
            [],
            id="a-checkers-own-table-in-the-project-ends-the-search",
        ),
        pytest.param(
            {"ws/pyproject.toml": "[tool.mypy]\n", "ws/project/pyproject.toml": "[tool.ty]\n"},
            ["ws/pyproject.toml"],
            id="a-pyproject-counts-for-the-checker-whose-table-it-holds",
        ),
        pytest.param(
            {"ws/setup.cfg": "[flake8]\nmax-line-length = 99\n"},
            [],
            id="a-setup-cfg-without-a-mypy-section-is-passed-over",
        ),
        pytest.param(
            {"ws/setup.cfg": "[mypy]\nstrict = true\n"},
            ["ws/setup.cfg"],
            id="a-setup-cfg-with-a-mypy-section-counts",
        ),
        pytest.param(
            {"ruff.toml": "", "pyrefly.toml": ""},
            ["pyrefly.toml", "ruff.toml"],
            id="the-others-follow-ruff-up-to-the-folder-it-reached",
        ),
        pytest.param(
            {"pyrefly.toml": ""},
            [],
            id="a-file-above-the-parent-is-left-when-ruff-does-not-reach-it",
        ),
        pytest.param(
            {
                "ws/project/ruff.toml": 'extend = "../one.toml"\n',
                "ws/one.toml": 'extend = "two.toml"\n',
                "ws/two.toml": "",
            },
            ["ws/one.toml", "ws/two.toml"],
            id="ruff-follows-a-chain-of-extensions",
        ),
        pytest.param(
            {
                "ws/project/ruff.toml": 'extend = "../a.toml"\n',
                "ws/a.toml": 'extend = "b.toml"\n',
                "ws/b.toml": 'extend = "a.toml"\n',
            },
            ["ws/a.toml", "ws/b.toml"],
            id="a-loop-of-extensions-ends-where-it-closes",
        ),
        pytest.param(
            {
                "ws/project/.ruff.toml": 'extend = "../one.toml"\n',
                "ws/project/ruff.toml": 'extend = "../two.toml"\n',
                "ws/one.toml": "",
                "ws/two.toml": "",
            },
            ["ws/one.toml"],
            id="the-file-ruff-prefers-in-a-folder-is-the-only-one-followed",
        ),
        pytest.param(
            {"ws/ruff.toml": "= broken", "ruff.toml": ""},
            ["ruff.toml"],
            id="a-file-that-does-not-parse-is-passed-over",
        ),
    ],
)
def test_each_checker_reads_the_nearest_file_that_configures_it(
    tmp_path: Path, files: dict[str, str], outside: list[str]
) -> None:
    read = _read(tmp_path, files)

    held = sorted(file.relative_to(tmp_path.resolve()).as_posix() for file in read.outside)
    assert held == outside


@pytest.mark.parametrize(
    ("files", "named", "root"),
    [
        pytest.param(
            {
                "ws/project/ty.toml": '[environment]\nextra-paths = ["..", "../stubs"]\n',
                "ws/stubs/s.pyi": "",
            },
            ["ws", "ws/stubs"],
            "ws",
            id="a-search-root-above-the-project-comes-before-the-folder-in-it",
        ),
        pytest.param(
            {"ws/project/ty.toml": _UNNAMED, "ws/project/src/x.py": ""},
            [],
            "ws",
            id="the-project-itself-an-absolute-path-and-a-missing-one-name-nothing",
        ),
        pytest.param(
            {
                "ws/project/ruff.toml": 'extend = "../base.toml"\n',
                "ws/base.toml": 'src = ["stubs"]\n',
                "ws/stubs/s.pyi": "",
            },
            ["ws/stubs"],
            "ws",
            id="an-extended-file-names-places-from-its-own-folder",
        ),
        pytest.param(
            {
                "ws/project/ty.toml": '[environment]\nextra-paths = ["../../stubs"]\n',
                "stubs/s.pyi": "",
            },
            ["stubs"],
            ".",
            id="a-place-above-the-parent-raises-the-root",
        ),
        pytest.param(
            {
                "ws/project/mypy.ini": "[mypy]\nmypy_path = ../a:../b,../c\n",
                "ws/a/x": "",
                "ws/b/x": "",
                "ws/c/x": "",
            },
            ["ws/a", "ws/b", "ws/c"],
            "ws",
            id="an-ini-value-lists-its-places-by-colon-and-comma",
        ),
    ],
)
def test_the_places_a_setting_names_are_those_that_exist_outside_the_project(
    tmp_path: Path, files: dict[str, str], named: list[str], root: str
) -> None:
    read = _read(tmp_path, files)

    base = tmp_path.resolve()
    held = [place.relative_to(base).as_posix() for place in read.named]
    assert (held, read.root.relative_to(base).as_posix()) == (named, root)


@settings(max_examples=25)
@given(listed=st.lists(st.sampled_from([*_PLACES, "missing"]), max_size=6))
@example(listed=["gamma", "alpha", "missing", "alpha"])
def test_the_places_named_are_the_listed_ones_that_exist_shortest_path_first(
    listed: list[str],
) -> None:
    with TemporaryDirectory() as scratch:
        base = Path(scratch).resolve()
        paths = json.dumps([f"../{name}" for name in listed])
        files = {f"ws/{name}/x": "" for name in _PLACES}
        read = _read(
            base, {"ws/project/ty.toml": f"[environment]\nextra-paths = {paths}\n", **files}
        )

    assert read.named == sorted(base / "ws" / name for name in set(listed) & set(_PLACES))
