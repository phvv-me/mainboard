from importlib.metadata import PackageNotFoundError
from pathlib import Path
from types import SimpleNamespace

import pytest

from mainboard import MissionError, Project
from mainboard.core import project as project_module

# The names an install of this checkout declares, spelled out so a test reads as the layout it
# checks; the first test holds them equal to what the installed metadata says.
_NAMES = ("mb", "mainboard")


def test_every_name_derives_from_the_console_scripts_the_install_declares(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The shortest script pointing at the CLI is the primary name, the rest legacy names."""
    project = Project()
    assert (project.names, project.name, project.package) == (_NAMES, "mb", "mainboard")
    assert project.manifests == ("mb.toml", "mainboard.toml")
    assert project.out_dirs == (".mb", ".mainboard")
    assert project.jobs_roots == ("~/.mb-jobs", "~/.mainboard-jobs")
    assert project.plugin_groups == ("mb.providers", "mainboard.providers")

    def scripts(*declared: tuple[str, str]) -> SimpleNamespace:
        points = [SimpleNamespace(name=name, value=value) for name, value in declared]
        return SimpleNamespace(entry_points=SimpleNamespace(select=lambda group: points))

    declared = scripts(
        ("mainboard", "mainboard.cli:main"),
        ("zz", "mainboard.cli:main"),
        ("mb", "mainboard.cli:main"),
        ("m", "other.cli:main"),
    )
    monkeypatch.setattr(project_module, "distribution", lambda name: declared)
    assert project_module._names() == ("mb", "zz", "mainboard")
    monkeypatch.setattr(project_module, "distribution", lambda name: scripts())
    assert project_module._names() == ("mainboard",)

    def uninstalled(name: str) -> None:
        raise PackageNotFoundError(name)

    monkeypatch.setattr(project_module, "distribution", uninstalled)
    assert project_module._names() == ("mainboard",)


def test_a_workspace_is_found_under_either_manifest_name_and_never_under_both(
    tmp_path: Path,
) -> None:
    """Discovery walks upward accepting any name, and one directory holding two is refused."""
    project = Project(names=_NAMES)
    legacy, current = tmp_path / "legacy", tmp_path / "current"
    (legacy / "a" / "b").mkdir(parents=True)
    (legacy / "mainboard.toml").write_text("")
    current.mkdir()
    (current / "mb.toml").write_text("")
    assert project.find_root(legacy / "a" / "b") == legacy
    assert project.manifest(legacy) == legacy / "mainboard.toml"
    assert project.find_root(current) == current
    assert project.manifest(current) == current / "mb.toml"
    # A directory with no manifest gets the primary name, which is where a new one goes.
    assert project.manifest(tmp_path) == tmp_path / "mb.toml"
    (legacy / "mb.toml").write_text("")
    with pytest.raises(MissionError, match=r"both mb\.toml and mainboard\.toml.*keep mb\.toml"):
        project.find_root(legacy / "a")
    orphan = tmp_path.parent / f"{tmp_path.name}-orphan"
    orphan.mkdir()
    with pytest.raises(FileNotFoundError, match="no missing.toml found .* inside a workspace"):
        Project(names=("missing",)).find_root(orphan)
    assert Project(names=("missing",)).workspace(orphan) == orphan


def test_a_workspace_keeps_the_state_directory_it_has_and_a_fresh_one_follows_its_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nothing is renamed: an existing directory wins, and with none the manifest's name does."""
    project = Project(names=_NAMES)
    fresh, legacy, clone = tmp_path / "fresh", tmp_path / "legacy", tmp_path / "clone"
    for root, manifest in ((fresh, "mb.toml"), (legacy, "mainboard.toml"), (clone, "mb.toml")):
        root.mkdir()
        (root / manifest).write_text("")
    (legacy / ".mainboard").mkdir()
    (clone / ".mainboard").mkdir()
    assert project.out_dir(fresh) == ".mb"
    assert project.out_dir(legacy) == ".mainboard"
    # A legacy state directory is kept whatever the manifest is now called.
    assert project.out(clone) == clone / ".mainboard"
    # A legacy manifest with no state yet (a clone, a host's mirror) lands where its origin does.
    (legacy / ".mainboard").rmdir()
    assert project.out_dir(legacy) == ".mainboard"
    # Once the primary directory exists it wins.
    (clone / ".mb").mkdir()
    assert project.out_dir(clone) == ".mb"
    assert project.out_dir(tmp_path) == ".mb"
    monkeypatch.chdir(legacy)
    assert (project.out_dir(), project.out()) == (".mainboard", legacy / ".mainboard")
    assert project.activation() == ".mainboard/envs/default/activate.sh"
    assert project.activation("serving", fresh) == ".mb/envs/serving/activate.sh"


def test_a_workspace_keeps_the_lock_it_has_and_a_fresh_one_takes_the_primary_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The lock is tracked and travels itself, so a legacy manifest does not pull it along."""
    project = Project(names=_NAMES)
    assert project.locks == ("mb.lock", "mainboard.lock")
    (tmp_path / "mainboard.toml").write_text("")
    assert project.lock(tmp_path) == tmp_path / "mb.lock"
    monkeypatch.chdir(tmp_path)
    assert project.lock() == tmp_path / "mb.lock"
    (tmp_path / "mainboard.lock").write_text("")
    assert project.lock(tmp_path) == tmp_path / "mainboard.lock"
    (tmp_path / "mb.lock").write_text("")
    with pytest.raises(MissionError, match=r"both mb\.lock and mainboard\.lock.*keep mb\.lock"):
        project.lock(tmp_path)


def test_a_variable_is_read_under_either_name_and_exported_under_both() -> None:
    """A job may import an older release than the one that dispatched it."""
    source = Project(names=_NAMES).variable("SOURCE")
    assert source.names == ("MB_SOURCE", "MAINBOARD_SOURCE")
    assert source.exported("v1") == {"MB_SOURCE": "v1", "MAINBOARD_SOURCE": "v1"}
    assert source.read({"MAINBOARD_SOURCE": "old"}) == "old"
    assert source.read({"MAINBOARD_SOURCE": "old", "MB_SOURCE": "new"}) == "new"
    assert (source.read({}), source.present({})) == ("", False)
    assert source.present({"MAINBOARD_SOURCE": ""})


def test_markers_are_written_under_the_legacy_name_and_read_under_any(tmp_path: Path) -> None:
    """Center and hosts may run different releases, so the name every one reads is written."""
    project = Project(names=_NAMES)
    assert project.marker("prefix") == ".mainboard-prefix"
    assert project.markers("prefix") == (".mb-prefix", ".mainboard-prefix")
    assert project.marked(tmp_path, "prefix") == tmp_path / ".mainboard-prefix"
    (tmp_path / ".mb-prefix").write_text("")
    assert project.marked(tmp_path, "prefix") == tmp_path / ".mb-prefix"
    tool = {"mainboard": {"ci": "legacy"}}
    assert project.table(tool) == {"ci": "legacy"}
    assert project.table({**tool, "mb": {"ci": "current"}}) == {"ci": "current"}
    assert project.table({"ruff": {}}) is None
