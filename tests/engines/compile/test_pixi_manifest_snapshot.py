import platform
import tomllib
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from mainboard import Project, load
from mainboard.engines.compile.pixi_manifest import PixiManifest

if TYPE_CHECKING:
    from mainboard.manifest import Manifest

_PROJECT = Project().name
_FIXTURES = Path(__file__).parent / "fixtures"


def test_the_whole_compile_surface_renders_the_pinned_pixi_manifest(
    manifest_from: Callable[[str], Manifest],
) -> None:
    """The kitchen manifest separates default, inherited and isolated solve surfaces.

    `fixtures/kitchen.toml` declares every table shape the compiler emits, so a change in
    what pixi is handed shows up as a diff.

    It carries a workspace floor one environment raises, dependency sources that must be
    rerooted out of `.mainboard/` and one that must not, tasks in all four shapes, solver
    options with their override sub-table, per-platform overlays, a dev table, and an
    environment that starts from nothing but itself.
    """
    source = (_FIXTURES / "kitchen.toml").read_text(encoding="utf-8")
    manifest = manifest_from(source)
    default = tomllib.loads(
        PixiManifest.from_manifest(manifest, project_name=_PROJECT).to_toml()
    )
    serving = tomllib.loads(
        PixiManifest.from_manifest(
            manifest, project_name=_PROJECT, environment="serving"
        ).to_toml()
    )
    isolated = tomllib.loads(
        PixiManifest.from_manifest(
            manifest, project_name=_PROJECT, environment="isolated"
        ).to_toml()
    )

    assert set(default["dependencies"]) == {"python", "path"}
    assert set(default["feature"]) == {"mainboard-platforms", "dev"}
    assert "serving" not in default["feature"]
    assert set(serving["dependencies"]) == {"python", "path"}
    assert set(serving["feature"]) == {"serving"}
    assert serving["workspace"]["platforms"] == [
        {"name": "linux-64-serving", "platform": "linux-64", "cuda": "13.0"}
    ]
    assert "dependencies" not in isolated
    assert set(isolated["feature"]) == {"isolated"}
    assert isolated["environments"]["isolated"]["no-default-feature"] is True


def test_the_compiler_reads_nothing_that_belongs_to_another_subsystem(workspace: Path) -> None:
    """The compiler stays inside its own lane.

    The `workspace` fixture is the repo's one full-featured manifest, carrying `[vars]`
    interpolation, `[hosts.*]` and `[containers.*]`, none of which the pixi compiler reads.
    """
    manifest = load(workspace / Project().manifest)
    compiled = PixiManifest.from_manifest(manifest, project_name=_PROJECT)
    document = tomllib.loads(compiled.to_toml())

    # The default shard carries only the root platforms; serving's CUDA variants live in its
    # own shard and cannot force the default lock to solve them.
    assert document["workspace"]["platforms"] == [
        "linux-64",
        "linux-aarch64",
    ]
    assert document["activation"] == {"scripts": ["dotenv.sh"]}
    assert "serving" not in document.get("feature", {})
    serving = tomllib.loads(
        PixiManifest.from_manifest(
            manifest, project_name=_PROJECT, environment="serving"
        ).to_toml()
    )
    assert serving["feature"]["serving"]["platforms"] == [
        "linux-64-serving",
        "linux-aarch64-serving",
    ]
    # `[vars]` interpolation landed (`station` mixes `os_name()`/`arch()`), but only inside the
    # source manifest fields the compiler actually reads, never as a table of its own.
    family = {"Darwin": "macos", "Windows": "windows"}.get(platform.system(), "linux")
    assert manifest.vars["station"] == f"{family}-{platform.machine()}"
    for leaked in ("hosts", "containers", "vars", "ngc", "miyabi"):
        assert leaked not in compiled.to_toml()
