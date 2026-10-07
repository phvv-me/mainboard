"""A locked local wheel must travel with the environment that installs it."""

from pathlib import Path, PurePosixPath

from mainboard.engines.compile.pixi_manifest import local_sources, self_installed
from mainboard.engines.compile.provisioner import Provisioner
from mainboard.manifest import Manifest


def test_local_files_are_shipped_without_becoming_editable_roots(tmp_path: Path) -> None:
    wheel = tmp_path / "wheels/example-1.0-py3-none-any.whl"
    wheel.parent.mkdir()
    wheel.write_bytes(b"wheel fixture")
    manifest = Manifest.model_validate(
        {
            "workspace": {"name": "wheel-test", "platforms": ["linux-64"]},
            "python": {"deps": {"example": {"path": "wheels/" + wheel.name}}},
        }
    )
    provisioner = Provisioner(tmp_path, manifest)
    provisioner.recompiled()
    assert "wheels/" + wheel.name in provisioner.artifact
    # Linux raises NotADirectoryError for wheel/pyproject.toml; no metadata is owed there.
    assert len(provisioner.compiler.resolution_digest()) == 64
    compiled = provisioner.pixi.manifest.read_text()
    directory = provisioner.environment_dir().relative_to(tmp_path)
    assert local_sources(compiled, generated_dir=directory) == ["wheels/" + wheel.name]
    assert self_installed(compiled, generated_dir=directory) == []


def test_sources_in_features_and_targets_are_deduplicated() -> None:
    manifest = """[pypi-dependencies]
example = {path = "../../../wheels/example.whl"}
local = {path = "../../../packages/local", editable = true}
[feature.extra.target.linux-64.pypi-dependencies]
example = {path = "../../../wheels/example.whl"}
external = {path = "/outside/example.whl"}
"""
    directory = PurePosixPath(".mainboard/envs/default")
    assert local_sources(manifest, generated_dir=directory) == [
        "wheels/example.whl",
        "packages/local",
    ]
    assert self_installed(manifest, generated_dir=directory) == ["packages/local"]
