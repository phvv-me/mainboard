import platform
import tomllib
from pathlib import Path
from typing import TYPE_CHECKING

from mainboard import Project, load
from mainboard.engines.compile.pixi_manifest import PixiManifest

if TYPE_CHECKING:
    from collections.abc import Callable

    from mainboard.manifest import Manifest

_PROJECT = Project().name
_FIXTURES = Path(__file__).parent / "fixtures"


def test_the_whole_compile_surface_renders_the_pinned_pixi_manifest(
    manifest_from: Callable[[str], Manifest],
) -> None:
    """`fixtures/kitchen.toml` declares every table shape the compiler emits, and its rendered
    pair is pinned as text so a change in what pixi is handed shows up as a diff.

    It carries a workspace floor one environment raises, dependency sources that must be
    rerooted out of `.mainboard/` and one that must not, tasks in all four shapes, solver
    options with their override sub-table, per-platform overlays, a dev table, and an
    environment that starts from nothing but itself.
    """
    source = (_FIXTURES / "kitchen.toml").read_text(encoding="utf-8")
    compiled = PixiManifest.from_manifest(manifest_from(source), project_name=_PROJECT)
    assert compiled.to_toml() == (_FIXTURES / "kitchen.pixi.toml").read_text(encoding="utf-8")


def test_the_compiler_reads_nothing_that_belongs_to_another_subsystem(workspace: Path) -> None:
    """`tests/conftest.py`'s `workspace` fixture is the repo's one full-featured manifest,
    carrying `[vars]` interpolation, `[hosts.*]` and `[containers.*]`, none of which the pixi
    compiler reads, so this is the proof that it stays inside its own lane.
    """
    manifest = load(workspace / Project().manifest)
    compiled = PixiManifest.from_manifest(manifest, project_name=_PROJECT)
    document = tomllib.loads(compiled.to_toml())

    # The root platforms ride bare, since no floor is declared workspace-wide, and `serving`'s
    # own `cuda` floor adds one named variant per platform beside them.
    assert document["workspace"]["platforms"] == [
        "linux-64",
        "linux-aarch64",
        {"name": "linux-64-serving", "platform": "linux-64", "cuda": "13.0"},
        {"name": "linux-aarch64-serving", "platform": "linux-aarch64", "cuda": "13.0"},
    ]
    assert document["activation"] == {"scripts": ["dotenv.sh"]}
    assert document["feature"]["serving"]["platforms"] == [
        "linux-64-serving",
        "linux-aarch64-serving",
    ]
    # `[vars]` interpolation landed (`station` mixes `os_name()`/`arch()`), but only inside the
    # source manifest fields the compiler actually reads, never as a table of its own.
    assert manifest.vars["station"] == f"linux-{platform.machine()}"
    for leaked in ("hosts", "containers", "vars", "ngc", "miyabi"):
        assert leaked not in compiled.to_toml()
