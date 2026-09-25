from pathlib import Path

from mainboard.dispatch.shared import workspace


def test_a_directory_under_no_workspace_keeps_its_own_state(tmp_path: Path) -> None:
    assert workspace(tmp_path) == tmp_path
