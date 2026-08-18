import platform
import tomllib
from typing import TYPE_CHECKING

from mainboard import Project, load
from mainboard.engines.compile.pixi_manifest import PixiManifest

if TYPE_CHECKING:
    from pathlib import Path

_PROJECT = Project().name


def test_from_manifest_snapshot_of_the_shared_fixture_workspace(workspace: Path) -> None:
    """Everything hosts/containers/vars/interpolation touch stays out of the compiled manifest.

    `tests/conftest.py`'s `workspace` fixture is the repo's one full-featured manifest, carrying
    `[vars]` interpolation, `[hosts.*]`, and `[containers.*]`, none of which the pixi compiler
    reads, so this snapshot is also the proof that the compiler stays inside its own lane.
    """
    manifest = load(workspace / Project().manifest)
    compiled = PixiManifest.from_manifest(manifest, project_name=_PROJECT)
    document = tomllib.loads(compiled.to_toml())

    assert document["workspace"]["name"] == "lab"
    assert document["workspace"]["version"] == "0.1.0"
    assert document["workspace"]["channels"] == ["conda-forge"]
    # The root platforms ride bare, and `serving`'s own `cuda` floor adds one named variant
    # per platform beside them.
    assert document["workspace"]["platforms"] == [
        "linux-64",
        "linux-aarch64",
        {"name": "linux-64-serving", "platform": "linux-64", "cuda": "13.0"},
        {"name": "linux-aarch64-serving", "platform": "linux-aarch64", "cuda": "13.0"},
    ]
    assert document["activation"] == {"scripts": ["dotenv.sh"]}
    assert document["dependencies"] == {"python": ">=3.14", "pueue": "*"}
    assert document["pypi-dependencies"] == {
        "torch": ">=2.9",
        "lab-core": {"path": "../packages/lab-core", "editable": True},
    }
    assert document["tasks"] == {"test": {"cmd": "pytest", "cwd": "../packages/lab-core"}}

    # The `serving` env raises its own `cuda` floor (through `{{ vars.cuda }}`), which is what
    # forces a named platform variant instead of a bare platform string.
    serving = document["feature"]["serving"]
    assert serving["pypi-dependencies"] == {"vllm": "*"}
    assert serving["platforms"] == ["linux-64-serving", "linux-aarch64-serving"]

    # `[vars]` interpolation landed (`station` mixes `os_name()`/`arch()`), but only inside the
    # source manifest fields the compiler actually reads, never as a table of its own.
    assert manifest.vars["station"] == f"linux-{platform.machine()}"

    # Nothing from hosts, containers, or the raw `[vars]` table crosses into the compiled pixi
    # manifest, that subsystem boundary is the whole point of this snapshot.
    for leaked in ("hosts", "containers", "vars"):
        assert leaked not in document
    assert "ngc" not in compiled.to_toml()
    assert "miyabi" not in compiled.to_toml()
