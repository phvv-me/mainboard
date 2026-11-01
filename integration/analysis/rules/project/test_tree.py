"""`ProjectTree` lists what a project's `.gitignore` files keep, however its root is spelled."""

from pathlib import Path

import pytest

from mainboard.analysis.rules.families.project.core import ProjectTree

_KEPT = [(".gitignore", False), ("kept.py", False)]


def test_a_root_spelled_relatively_is_judged_as_its_absolute_spelling_is(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    for name, text in {".gitignore": "ignored.py\n", "ignored.py": "", "kept.py": ""}.items():
        (project / name).write_text(text, encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    relative = ProjectTree(root=Path("project")).entries()

    assert relative == ProjectTree(root=project).entries() == _KEPT
