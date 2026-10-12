"""A no-default environment carries its own PyPI dependency-overrides, and `update` installs with
editable sources.

Two fixes exercised here: `[envs.<name>.python.dependency-overrides]` reaches that environment's
feature (previously only the root override reached `default`), and `Pixi.update` drops pixi's
`--no-install` when the lock has an editable source (which pixi must build to read, so the flag
would fail the solve).
"""

from pathlib import Path

import pytest

from mainboard.engines.compile.backend.pixi import Pixi
from mainboard.engines.compile.pixi_manifest import PixiManifest, pypi_options
from mainboard.manifest import Manifest
from mainboard.manifest.members import Composition


def _workspace_with_lean_env(root: Path) -> Manifest:
    """A root with one editable member and a `no-default` env declaring only an override."""
    (root / "pkg").mkdir()
    (root / "pkg" / "pyproject.toml").write_text(
        '[project]\nname = "pkg"\nversion = "0.0.0"\nrequires-python = ">=3.14"\n',
        encoding="utf-8",
    )
    (root / "pkg" / "mainboard.toml").write_text('[workspace]\nname = "pkg"\n', encoding="utf-8")
    (root / "mainboard.toml").write_text(
        '[workspace]\nname = "it"\nmembers = ["pkg"]\n\n'
        '[deps]\npython = ">=3.14"\n\n'
        '[python.deps]\npkg = { path = "pkg", editable = true }\n\n'
        '[envs.x]\nno-default = true\nplatforms = ["linux-64"]\n\n'
        '[envs.x.python.deps]\npkg = { path = "pkg", editable = true }\n\n'
        '[envs.x.python.dependency-overrides]\nnvidia-cublas = ">=13.8.1.7,<14"\n',
        encoding="utf-8",
    )
    return Manifest.model_validate(
        __import__("tomllib").loads((root / "mainboard.toml").read_text())
    )


def test_no_default_env_override_reaches_its_feature(tmp_path: Path) -> None:
    composed = Composition(tmp_path, _workspace_with_lean_env(tmp_path)).composed()

    options = pypi_options(composed.envs["x"])
    assert options["dependency-overrides"]["nvidia-cublas"] == ">=13.8.1.7,<14"

    shard = PixiManifest.from_manifest(composed, project_name="it", environment="x")
    feature = shard.model_dump(mode="python", round_trip=True)["feature"]["x"]
    override = feature["pypi-options"]["dependency-overrides"]
    assert override["nvidia-cublas"] == ">=13.8.1.7,<14"
    assert "path" in override["pkg"], (
        "the member path-pin must still be unioned under the override"
    )


class _Ran:
    returncode = 0


@pytest.mark.parametrize(("editable", "expected"), [(True, False), (False, True)])
def test_update_drops_no_install_only_without_editable_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, editable: bool, expected: bool
) -> None:
    pixi = Pixi(tmp_path)
    seen: list[tuple[str, ...]] = []
    monkeypatch.setattr(pixi, "_has_editable_paths", lambda: editable)
    monkeypatch.setattr(
        pixi, "within_cwd", lambda action, *args, **flags: seen.append(args) or _Ran()
    )

    pixi.update("default")

    assert ("--no-install" in seen[0]) is expected
    assert seen[0][0] == "update" and "default" in seen[0]
