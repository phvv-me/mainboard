from pathlib import Path

import pytest

from mainboard import Project


def test_every_name_derives_from_the_package() -> None:
    project = Project()
    assert project.manifest == f"{project.name}.toml"
    assert project.out_dir == f".{project.name}"
    assert project.plugin_group == f"{project.name}.providers"


def test_activation_gives_every_environment_but_the_default_its_own_script() -> None:
    project = Project()
    assert project.activation() == f"{project.out_dir}/activate.sh"
    assert project.activation("serving") == f"{project.out_dir}/activate-serving.sh"


def test_find_root_walks_up_and_refuses_a_rootless_tree(tmp_path: Path) -> None:
    project = Project()
    (tmp_path / project.manifest).write_text("")
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)
    assert project.find_root(nested) == tmp_path
    orphan = tmp_path.parent / f"{tmp_path.name}-orphan"
    orphan.mkdir()
    with pytest.raises(FileNotFoundError, match="run inside a workspace"):
        project.find_root(orphan)
