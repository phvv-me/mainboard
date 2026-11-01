"""A `Workspace` names a project's packages and submodules from one listing of its files."""

from collections.abc import Callable
from pathlib import Path

import pytest

from mainboard.analysis.project.workspace import Workspace

from ..engine.builders import written

_ONE_LISTING = ["ls-files", "ls-files", "check-attr"]
_SUBMODULES = frozenset({"src/pkg/mod", "pkg/mod", "mod", "src/pkg", "pkg", "a/x", "x"})
_OWNERS = '[lint]\nowners = ["libs/*"]\n'


def test_a_workspace_lists_its_project_once_for_the_packages_and_the_submodules_it_names(
    repository: Path, git_verbs: Callable[[], list[str]]
) -> None:
    written(repository, {"src/pkg/__init__.py": "", "src/pkg/mod.py": ""})
    workspace = Workspace(repository)

    named = (workspace.packages(), workspace.submodules())

    assert (named, git_verbs()) == ((["pkg"], _SUBMODULES), _ONE_LISTING)


@pytest.mark.parametrize(
    ("initializer", "module"),
    [
        pytest.param("__init__.py", "tool.py", id="source-files"),
        pytest.param("__init__.pyi", "tool.pyi", id="stub-files"),
    ],
)
def test_a_package_and_a_module_at_the_source_root_are_named_whether_they_are_source_or_stub(
    repository: Path, initializer: str, module: str
) -> None:
    written(repository, {f"src/pkg/{initializer}": "", f"src/{module}": "", "src/conftest.py": ""})

    assert Workspace(repository).packages() == ["pkg", "tool"]


@pytest.mark.parametrize(
    "initializer",
    [
        pytest.param("__init__.py", id="an-owner-holding-a-source-package"),
        pytest.param("__init__.pyi", id="an-owner-holding-a-stub-package"),
    ],
)
def test_an_owner_holding_a_package_of_the_project_is_no_consumer_with_a_stub_initializer_too(
    repository: Path, initializer: str
) -> None:
    written(
        repository,
        {
            "mb.toml": _OWNERS,
            "project/src/pkg/__init__.py": "",
            "libs/user/app.py": "import pkg\n",
            f"libs/beside/pkg/{initializer}": "",
            f"libs/below/src/pkg/{initializer}": "",
        },
    )

    assert Workspace(repository / "project").consumers(["pkg"]) == ["libs/user"]
