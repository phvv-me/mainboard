"""PY-CONS0001's fix renames a public constant only where no star import can read it."""

from pathlib import Path

import pytest

from ....builders import fixed

_CONSTANT = "LIMIT = 3\n\n\ndef clamp(value):\n    return min(value, LIMIT)\n"


@pytest.mark.parametrize(
    "files",
    [
        pytest.param(
            {"src/pkg/mod.py": _CONSTANT, "src/pkg/user.py": "from pkg.mod import *\n"},
            id="a-star-import-of-a-source-module",
        ),
        pytest.param(
            {"src/pkg/mod.pyi": _CONSTANT, "src/pkg/user.py": "from pkg.mod import *\n"},
            id="a-star-import-of-a-stub-module",
        ),
        pytest.param(
            {"src/pkg/__init__.py": _CONSTANT, "src/user.py": "from pkg import *\n"},
            id="a-star-import-of-a-source-package",
        ),
        pytest.param(
            {"src/pkg/__init__.pyi": _CONSTANT, "src/user.py": "from pkg import *\n"},
            id="a-star-import-of-a-stub-package",
        ),
    ],
)
def test_a_constant_a_star_import_reads_stays_public_in_source_and_in_stub(
    tmp_path: Path, files: dict[str, str]
) -> None:
    project = {"src/pkg/__init__.py": "", **files}

    assert fixed(tmp_path, project, rules=("PY-CONS0001",)) == project
