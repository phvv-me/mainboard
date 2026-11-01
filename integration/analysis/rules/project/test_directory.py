"""The `Directory` family counts a package's files and modules and places its source root the
same for a stub initializer as for a source one."""

from pathlib import Path

import pytest

from mainboard.analysis.engine.lint import Lint, load_catalog
from mainboard.analysis.rules.general.project.filesystem import DirectoryModuleCount

from ...engine.builders import written

_ROWS = """
    SELECT path, direct_file_count, direct_module_count, source_depth
    FROM directory_facts WHERE starts_with(path, 'lib') ORDER BY path
"""
# The initializer is no file of `lib/pkg`, and `lib` is the source root above the package.
_PACKAGE = [("lib", 0, 0, 0), ("lib/pkg", 1, 1, 1), ("lib/pkg/sub", 1, 1, 2)]


@pytest.mark.parametrize(
    "initializer",
    [
        pytest.param("__init__.py", id="a-source-initializer"),
        pytest.param("__init__.pyi", id="a-stub-initializer"),
    ],
)
def test_a_package_initializer_is_no_file_of_its_directory_and_marks_the_source_root_above(
    tmp_path: Path, initializer: str
) -> None:
    root = written(
        tmp_path,
        {
            "pyproject.toml": "[tool.mcmr]\n",
            f"lib/pkg/{initializer}": "",
            "lib/pkg/a.py": "",
            "lib/pkg/sub/b.py": "",
        },
    )
    load_catalog()

    with Lint(root, (DirectoryModuleCount,), fixing=False) as lint:
        rows = lint.facts(root).cursor(None).execute(_ROWS).fetchall()

    assert rows == _PACKAGE
