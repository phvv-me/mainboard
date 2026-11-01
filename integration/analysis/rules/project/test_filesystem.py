from collections.abc import Mapping
from pathlib import Path

import pytest

from mainboard.analysis.engine.lint import Lint, Selection, fix, load_catalog
from mainboard.analysis.rules.general.project.filesystem import (
    DirectoryModuleCount,
    DirectoryPathway,
    PrefixedModuleGroup,
)

_HOOK = "class Backend:\n    def __init_subclass__(cls) -> None:\n        pass\n"
_PYDANTIC_HOOK = """class Backend:
    @classmethod
    def __pydantic_init_subclass__(cls) -> None:
        pass
"""
_PLAIN = "class Backend:\n    pass\n"
_TRACERS = """\
from .constants import K


def test_the_constant_is_read_beside_the_tracers():
    assert K
"""
_PROJECT = {
    "pyproject.toml": '[project]\nname = "pkg"\nversion = "0"\n',
    "src/pkg/__init__.py": "",
}


def _plugins(base: str, **changes: str) -> dict[str, str]:
    """A folder of a base and six modules of two subclasses each, `changes` replacing some."""
    modules = {"base": base} | {
        f"{vendor}_{part}": (
            f"from .base import Backend\n\n\nclass {vendor.title()}{part.title()}(Backend):\n"
            f"    pass\n\n\nclass {vendor.title()}{part.title()}Spec(Backend):\n    pass\n"
        )
        for vendor in ("vast", "lambda", "verda")
        for part in ("api", "backend")
    }
    return {
        f"src/pkg/plugins/{name}.py": changes.get(name, text) for name, text in modules.items()
    }


def _written(root: Path, files: Mapping[str, str]) -> None:
    for name, content in {**_PROJECT, **files}.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def _dissolved(root: Path, files: Mapping[str, str]) -> tuple[dict[str, str], list[str]]:
    """The files and the folders ALL-FILE0004's fix leaves of the project `files` make."""
    _written(root, files)
    load_catalog()
    fix(root, selection=Selection(rule=(DirectoryPathway.id,)), write=True, unsafe=True)
    entries = sorted(
        path.relative_to(root)
        for path in root.rglob("*")
        if not any(part.startswith(".") for part in path.relative_to(root).parts)
    )
    return (
        {
            entry.as_posix(): (root / entry).read_text(encoding="utf-8")
            for entry in entries
            if (root / entry).is_file()
        },
        [entry.as_posix() for entry in entries if (root / entry).is_dir()],
    )


def _flagged(root: Path, files: Mapping[str, str]) -> list[str]:
    _written(root, files)
    load_catalog()
    with Lint(root, (DirectoryModuleCount, PrefixedModuleGroup), fixing=False) as lint:
        return sorted(rule for rule, rows in lint.findings(lint.facts(root)).items() if rows)


@pytest.mark.parametrize(
    ("files", "flagged"),
    [
        pytest.param(_plugins(_HOOK), [], id="classes-an-init-subclass-hook-registers"),
        pytest.param(_plugins(_PYDANTIC_HOOK), [], id="classes-a-pydantic-hook-registers"),
        pytest.param(
            _plugins(_PLAIN),
            ["ALL-FILE0003", "ALL-FILE0005"],
            id="classes-nothing-registers",
        ),
        pytest.param(
            _plugins(_HOOK, verda_api="def helper() -> None:\n    pass\n"),
            ["ALL-FILE0003", "ALL-FILE0005"],
            id="a-module-that-defines-no-plugin",
        ),
        pytest.param(
            _plugins(_HOOK, verda_api="class Other:\n    pass\n"),
            ["ALL-FILE0003", "ALL-FILE0005"],
            id="a-class-outside-the-registry",
        ),
        pytest.param(
            _plugins(
                _HOOK, verda_api="def helper() -> None:\n    pass\n\n\nclass Other:\n    pass\n"
            ),
            ["ALL-FILE0003", "ALL-FILE0005"],
            id="a-function-beside-a-class",
        ),
        pytest.param(
            {
                f"src/pkg/plugins/{vendor}_{part}.py": f"def {vendor}_{part}() -> None: ...\n"
                for vendor in ("vast", "lambda", "verda")
                for part in ("api", "backend")
            },
            [],
            id="a-definition-catalog",
        ),
    ],
)
def test_a_folder_of_plugins_is_charged_no_modules_by_the_directory_ceilings(
    tmp_path: Path, files: Mapping[str, str], flagged: list[str]
) -> None:
    assert _flagged(tmp_path, files) == flagged


@pytest.mark.parametrize(
    ("before", "after", "folders"),
    [
        pytest.param(
            {
                "src/pkg/outer/__init__.py": "",
                "src/pkg/outer/inner/a.py": "A = 1\n",
                "src/pkg/app.py": "from .outer.inner.a import A\n\nB = A\n",
            },
            {
                "src/pkg/inner/a.py": "A = 1\n",
                "src/pkg/app.py": "from .inner.a import A\n\nB = A\n",
            },
            ["src", "src/pkg", "src/pkg/inner"],
            id="a-pathway-with-an-empty-initializer-gives-way-to-its-child",
        ),
        pytest.param(
            {
                "src/pkg/a/b/c/x.py": "X = 1\n",
                "src/pkg/app.py": "from .a.b.c.x import X\n\nY = X\n",
            },
            {
                "src/pkg/c/x.py": "X = 1\n",
                "src/pkg/app.py": "from .c.x import X\n\nY = X\n",
            },
            ["src", "src/pkg", "src/pkg/c"],
            id="a-chain-of-pathways-gives-way-to-the-last-child",
        ),
        pytest.param(
            {
                "tests/profile/providers/constants.py": "K = 1\n",
                "tests/profile/providers/test_tracers.py": _TRACERS,
            },
            {
                "tests/profile/constants.py": "K = 1\n",
                "tests/profile/test_tracers.py": _TRACERS,
            },
            ["src", "src/pkg", "tests", "tests/profile"],
            id="a-pathway-under-the-tests-root-keeps-its-name",
        ),
    ],
)
def test_the_fix_of_a_pathway_leaves_the_folders_where_the_rule_puts_them(
    tmp_path: Path, before: dict[str, str], after: dict[str, str], folders: list[str]
) -> None:
    files, found = _dissolved(tmp_path, before)

    assert files == {**_PROJECT, **after}
    assert found == folders
