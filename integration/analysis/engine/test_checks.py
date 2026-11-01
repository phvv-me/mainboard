"""`imported` names the source and stub files that a Python file's imports could mean."""

from pathlib import Path

import pytest

from mainboard.analysis.engine.checks import imported

from .builders import written


@pytest.mark.parametrize(
    "candidate",
    [
        pytest.param("src/pkg.py", id="a-source-module-under-src"),
        pytest.param("src/pkg.pyi", id="a-stub-module-under-src"),
        pytest.param("src/pkg/__init__.py", id="a-source-package-under-src"),
        pytest.param("src/pkg/__init__.pyi", id="a-stub-package-under-src"),
        pytest.param("app/pkg.pyi", id="a-stub-module-beside-the-file"),
        pytest.param("app/pkg/__init__.pyi", id="a-stub-package-beside-the-file"),
        pytest.param("pkg/__init__.pyi", id="a-stub-package-in-a-folder-above"),
    ],
)
def test_an_import_may_mean_a_source_or_a_stub_file_of_a_module_or_of_a_package(
    tmp_path: Path, candidate: str
) -> None:
    written(tmp_path, {"app/main.py": "import pkg\n"})

    assert candidate in imported(tmp_path, "app/main.py")
