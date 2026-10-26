from pathlib import Path

from mainboard.manifest import Sync


def test_an_include_naming_a_manifest_ships_the_one_the_workspace_holds(tmp_path: Path) -> None:
    (tmp_path / "mb.toml").write_text("", encoding="utf-8")
    (tmp_path / "mb.lock").write_text("", encoding="utf-8")

    shipped = Sync(include=["mainboard.toml", "scripts"]).shipped(tmp_path)

    assert shipped == ["mb.toml", "scripts", "mb.lock"]
