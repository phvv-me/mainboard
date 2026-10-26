"""The optional workspace test collector cannot become a CLI startup dependency."""

import os
import subprocess
import sys
from pathlib import Path

from mainboard.jobs.testing.lanes import parsed


def test_cli_help_does_not_import_pytest(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.modules['pytest'] = None; "
            "from mainboard.cli import build; build()(['app', '--help'])",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Build and test" in result.stdout


def test_workspace_collector_emits_cells_without_running_them(tmp_path: Path) -> None:
    target = tmp_path / "test_cells.py"
    target.write_text(
        """import pytest
@pytest.mark.parametrize('size', [1, 2])
def test_cell(size):
    raise AssertionError('collection must never execute a cell')
""",
        encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, "-m", "mainboard.jobs.testing.collector", "collect", str(target)],
        cwd=tmp_path,
        env={**os.environ, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"},
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    cells = parsed(result.stdout)
    assert [cell.key for cell in cells] == ["1", "2"]
    assert [cell.params for cell in cells] == [{"size": "1"}, {"size": "2"}]
