from pathlib import Path

import pytest

from mainboard import Project


def test_every_name_derives_from_the_package_and_every_environment_gets_its_own_script() -> None:
    """Renaming the tool is renaming the module, so nothing here spells the name literally."""
    project = Project()
    assert project.manifest == f"{project.name}.toml"
    assert project.out_dir == f".{project.name}"
    assert project.plugin_group == f"{project.name}.providers"
    assert project.activation() == f"{project.out_dir}/activate.sh"
    assert project.activation("serving") == f"{project.out_dir}/activate-serving.sh"


def test_find_root_walks_up_and_refuses_a_rootless_tree(tmp_path: Path) -> None:
    """A workspace is found from anywhere inside it, and its absence is said out loud."""
    project = Project()
    (tmp_path / project.manifest).write_text("")
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)
    assert project.find_root(nested) == tmp_path
    orphan = tmp_path.parent / f"{tmp_path.name}-orphan"
    orphan.mkdir()
    orphan_project = Project(name="missing-workspace")
    with pytest.raises(FileNotFoundError, match="run inside a workspace"):
        orphan_project.find_root(orphan)
