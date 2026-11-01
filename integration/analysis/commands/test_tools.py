"""The fix tools a person reads before writing anything: `preview` tells each rule's fix yield,
`trial` fixes a scratch copy that reads the Ruff configuration its project extends, and `move`
respells every import of the definitions it carries to another module."""

from collections.abc import Mapping
from pathlib import Path

import pytest

from mainboard.analysis.commands.tools import move, preview, trial
from mainboard.analysis.engine.lint import Selection
from mainboard.analysis.rules.python.libraries.logging import TextAroundTheLogger

# A line the default 88 columns would wrap, and the project's own 120 keeps whole.
_LONG = "kept whole, a line that Ruff's default width would wrap and the project's width keeps"
_LOGGING = Selection(rule=(TextAroundTheLogger.id,))


def _written(root: Path, files: Mapping[str, str]) -> None:
    for name, text in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text, encoding="utf-8")


def test_a_move_out_of_a_package_initializer_respells_each_importer_to_the_new_module(
    tmp_path: Path,
) -> None:
    _written(
        tmp_path,
        {
            "pyproject.toml": "[tool.mcmr]\n",
            "src/pkg/__init__.py": "",
            "src/pkg/a/__init__.py": "def helper() -> int:\n    return 1\n",
            "src/pkg/b.py": "from .a import helper\n\nVALUE = helper()\n",
        },
    )

    code = move(
        ("src/pkg/a/__init__.py::helper",),
        to=Path("src/pkg/a/helper.py"),
        root=tmp_path,
        write=True,
    )

    importer = (tmp_path / "src/pkg/b.py").read_text(encoding="utf-8")
    assert code == 0 and importer.startswith("from .a.helper import helper")


def test_a_preview_ends_with_what_each_rule_found_wrote_and_declined(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _written(
        tmp_path, {"pyproject.toml": "[tool.mcmr]\n", "printed.py": 'print("one")\nprint("two")\n'}
    )
    _written(tmp_path, {"declined.py": 'import sys\n\nprint("args", sys.argv)\n'})

    preview(tmp_path, selection=_LOGGING)

    shown = capsys.readouterr().out.splitlines()
    assert shown[-2:] == [
        "PY-LOGG0002: 3 found, 2 written, 1 declined, 0 deferred",
        "2 fixes planned, 0 deferred on a collision",
    ]


def test_a_trial_settles_its_fixes_by_the_configuration_its_project_extends(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    project, patch = tmp_path / "project", tmp_path / "fixes.patch"
    _written(tmp_path, {"pyproject.toml": "[tool.ruff]\nline-length = 120\n"})
    _written(
        project, {"pyproject.toml": '[tool.mcmr]\n\n[tool.ruff]\nextend = "../pyproject.toml"\n'}
    )
    _written(project, {"module.py": f'import sys\n\nARGS = sys.argv\nprint("{_LONG}")\n'})

    code = trial(project, selection=_LOGGING, patch=patch)

    shown = capsys.readouterr().out
    assert code == 0 and "settled the imports and format of 1 files" in shown
    assert f'+logger.result("{_LONG}")' in patch.read_text(encoding="utf-8")
