"""A no-default environment carries its own PyPI dependency-overrides, joins a member's environment
of its name, keeps the workspace's system floors, and `update` installs with editable sources.

Two fixes exercised here: `[envs.<name>.python.dependency-overrides]` reaches that environment's
feature (previously only the root override reached `default`), and `Pixi.update` drops pixi's
`--no-install` when the lock has an editable source (which pixi must build to read, so the flag
would fail the solve).
"""

import tomllib
from collections.abc import Mapping
from pathlib import Path

import pytest

from mainboard import Manifest
from mainboard.engines.compile.backend import Pixi
from mainboard.engines.compile.pixi.manifest import PixiManifest, pypi_options, selected_manifest
from mainboard.manifest.members import Composition


def _workspace_with_lean_env(root: Path) -> Manifest:
    """A root with one editable member and a `no-default` env declaring only an override."""
    (root / "pkg").mkdir()
    (root / "pkg" / "pyproject.toml").write_text(
        '[project]\nname = "pkg"\nversion = "0.0.0"\nrequires-python = ">=3.14"\n',
        encoding="utf-8",
    )
    (root / "pkg" / "mainboard.toml").write_text(
        """[workspace]
name = "pkg"

[envs.x]
platforms = ["osx-arm64"]

[envs.x.python.deps]
httpx = ">=0.28"

[envs.x.tasks]
serve = "python -m pkg"
""",
        encoding="utf-8",
    )
    (root / "mainboard.toml").write_text(
        """[workspace]
name = "it"
members = ["pkg"]

[system]
libc = "2.34"

[deps]
python = ">=3.14"

[python.deps]
pkg = { path = "pkg", editable = true }

[envs.x]
no-default = true
platforms = ["linux-64"]

[envs.x.python.deps]
pkg = { path = "pkg", editable = true }

[envs.x.python.dependency-overrides]
nvidia-cublas = ">=13.8.1.7,<14"
""",
        encoding="utf-8",
    )
    return Manifest.model_validate(tomllib.loads((root / "mainboard.toml").read_text()))


def test_no_default_env_override_reaches_its_feature(tmp_path: Path) -> None:
    composed = Composition(tmp_path, _workspace_with_lean_env(tmp_path)).composed()

    options = pypi_options(composed.envs["x"])
    overrides = options["dependency-overrides"]
    assert isinstance(overrides, Mapping) and overrides["nvidia-cublas"] == ">=13.8.1.7,<14"

    shard = PixiManifest.from_manifest(composed, project_name="it", environment="x")
    feature = shard.model_dump(mode="python", round_trip=True)["feature"]["x"]
    override = feature["pypi-options"]["dependency-overrides"]
    assert override["nvidia-cublas"] == ">=13.8.1.7,<14"
    assert "path" in override["pkg"], (
        "the member path-pin must still be unioned under the override"
    )


def test_member_env_joins_the_root_env_of_its_name(tmp_path: Path) -> None:
    x = Composition(tmp_path, _workspace_with_lean_env(tmp_path)).composed().envs["x"]

    assert x.no_default and x.platforms == ["linux-64"], "the root's settings win"
    assert {"pkg", "httpx"} <= x.model_dump(mode="python")["python"]["deps"].keys()
    assert "pkg:serve" in x.tasks, "a member's task keeps its member's namespace"


def test_lean_env_keeps_the_workspace_system_floors(tmp_path: Path) -> None:
    composed = Composition(tmp_path, _workspace_with_lean_env(tmp_path)).composed()

    assert selected_manifest(composed, "x").system == {"libc": "2.34"}


class _Ran:
    returncode = 0


@pytest.mark.parametrize(("editable", "expected"), [(True, False), (False, True)])
def test_update_drops_no_install_only_without_editable_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, editable: bool, expected: bool
) -> None:
    pixi = Pixi(tmp_path)
    seen: list[tuple[str, ...]] = []
    monkeypatch.setattr(pixi, "_has_editable_paths", lambda: editable)

    def within_cwd(action: str, *args: str, **flags: str) -> _Ran:
        seen.append(args)
        return _Ran()

    monkeypatch.setattr(pixi, "within_cwd", within_cwd)

    pixi.update("default")

    assert ("--no-install" in seen[0]) is expected
    assert seen[0][0] == "update" and "default" in seen[0]
