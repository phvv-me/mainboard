"""`module_files` names the modules of the runtime import graph: a stub is none, and its
initializer makes no package."""

from pathlib import Path

import pytest

from mainboard.analysis.commands.workspace import Workspace

from ...engine.builders import written

_ROWS = "SELECT path, module, packaged, initializer FROM module_files() ORDER BY path"


@pytest.mark.parametrize(
    ("initializer", "rows"),
    [
        pytest.param(
            "__init__.py",
            [
                {
                    "path": "pkg/__init__.py",
                    "module": "pkg",
                    "packaged": True,
                    "initializer": True,
                },
                {"path": "pkg/a.py", "module": "pkg.a", "packaged": True, "initializer": False},
            ],
            id="a-source-initializer-makes-a-package",
        ),
        pytest.param(
            "__init__.pyi",
            [{"path": "pkg/a.py", "module": "a", "packaged": False, "initializer": False}],
            id="a-stub-initializer-is-no-module-and-makes-no-package",
        ),
    ],
)
def test_only_a_source_initializer_makes_a_package_of_the_runtime_import_graph(
    tmp_path: Path, initializer: str, rows: list[dict[str, str | bool]]
) -> None:
    written(tmp_path, {f"pkg/{initializer}": "", "pkg/a.py": ""})

    assert Workspace(tmp_path).rows(_ROWS) == rows
