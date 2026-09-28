from pathlib import Path

import pytest

from mainboard.core.project import Project
from mainboard.profile import profiles_dir


def test_profiling_output_lands_under_the_workspaces_generated_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = Project()
    (project.manifest(tmp_path)).write_text("[workspace]\n", encoding="utf-8")
    nested = tmp_path / "research" / "bench"
    nested.mkdir(parents=True)
    monkeypatch.chdir(nested)
    assert profiles_dir("cutoken", "miyabi") == (
        project.out(tmp_path) / "profiles" / "cutoken" / "miyabi"
    )
    assert profiles_dir("cutoken", "miyabi").is_dir()
    assert profiles_dir(start=tmp_path) == project.out(tmp_path) / "profiles"
