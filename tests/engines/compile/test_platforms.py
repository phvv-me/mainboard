from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

import pytest

from mainboard import Manifest
from mainboard.core.host import current_platform
from mainboard.engines.compile.platforms import PlatformMatrix, SystemFloors
from mainboard.manifest import Header

if TYPE_CHECKING:
    from mainboard.engines.compile.toml import Toml

_LINUX64 = "linux-64"
_AARCH64 = "linux-aarch64"
_OSX = "osx-arm64"
_LINUX_ONLY = [_LINUX64, _AARCH64]
_BOTH_FAMILIES = [_LINUX64, _OSX]


@pytest.mark.parametrize(
    ("platforms", "declared", "workspace", "environments", "default"),
    [
        pytest.param(
            _LINUX_ONLY,
            "[envs.serving]\n",
            [_LINUX64, _AARCH64],
            {},
            [],
            id="no-floor-anywhere-leaves-platforms-bare-and-routes-nothing",
        ),
        pytest.param(
            _LINUX_ONLY,
            '[system]\ncuda = "13.0"\n[envs.serving]\n',
            [
                {"name": "linux-64-system", "platform": _LINUX64, "cuda": "13.0"},
                {"name": "linux-aarch64-system", "platform": _AARCH64, "cuda": "13.0"},
            ],
            {"serving": ["linux-64-system", "linux-aarch64-system"]},
            ["linux-64-system", "linux-aarch64-system"],
            id="a-workspace-wide-floor-names-every-platform-and-routes-every-env",
        ),
        pytest.param(
            _LINUX_ONLY,
            """[envs.serving]
system = { cuda = "13.0" }
[envs.cpu_only]
platforms = ["linux-aarch64"]
""",
            [
                _LINUX64,
                _AARCH64,
                {"name": "linux-64-serving", "platform": _LINUX64, "cuda": "13.0"},
                {"name": "linux-aarch64-serving", "platform": _AARCH64, "cuda": "13.0"},
            ],
            {
                "serving": ["linux-64-serving", "linux-aarch64-serving"],
                "cpu_only": [_AARCH64],
            },
            [_LINUX64, _AARCH64],
            id="one-env-raising-a-floor-names-the-variants-every-other-env-runs-on",
        ),
        pytest.param(
            _BOTH_FAMILIES,
            '[system]\nmacos = "14.0"\n[envs.serving]\n',
            [_LINUX64, {"name": "osx-arm64-system", "platform": _OSX, "macos": "14.0"}],
            {"serving": [_LINUX64, "osx-arm64-system"]},
            [_LINUX64, "osx-arm64-system"],
            id="a-macos-floor-reaches-the-osx-target-and-leaves-the-linux-one-bare",
        ),
        pytest.param(
            _LINUX_ONLY,
            '[system]\nmacos = "14.0"\n[envs.serving]\n',
            [_LINUX64, _AARCH64],
            {},
            [],
            id="a-macos-floor-with-no-osx-target-names-nothing-and-routes-nothing",
        ),
        pytest.param(
            _BOTH_FAMILIES,
            '[system]\ncuda = "13.0"\nglibc = "2.34"\nmacos = "14.0"\n[envs.serving]\n',
            [
                {
                    "name": "linux-64-system",
                    "platform": _LINUX64,
                    "cuda": "13.0",
                    "glibc": "2.34",
                },
                {"name": "osx-arm64-system", "platform": _OSX, "macos": "14.0"},
            ],
            {"serving": ["linux-64-system", "osx-arm64-system"]},
            ["linux-64-system", "osx-arm64-system"],
            id="each-platform-keeps-only-the-floors-its-own-family-can-provide",
        ),
        pytest.param(
            _BOTH_FAMILIES,
            '[envs.mac]\nsystem = { macos = "14.0" }\n',
            [_LINUX64, _OSX, {"name": "osx-arm64-mac", "platform": _OSX, "macos": "14.0"}],
            {"mac": [_LINUX64, "osx-arm64-mac"]},
            [_LINUX64, _OSX],
            id="an-env-raising-a-macos-floor-rides-the-bare-platform-where-it-reaches-nothing",
        ),
    ],
)
def test_a_virtual_package_floor_expands_into_named_platform_variants(
    platforms: Sequence[str],
    declared: str,
    workspace: list[Toml],
    environments: dict[str, list[str]],
    default: list[str],
    manifest_from: Callable[[str], Manifest],
) -> None:
    """A floor binds only the platforms whose family can provide it.

    Any floor that survives forces each env to name the variants it runs on.
    """
    listed = ", ".join(f'"{platform}"' for platform in platforms)
    manifest = manifest_from(f'[workspace]\nname = "w"\nplatforms = [{listed}]\n{declared}')
    matrix = PlatformMatrix.from_manifest(manifest)
    assert matrix.workspace == workspace
    assert matrix.environments == environments
    assert matrix.default == default


def test_a_floor_pixi_learns_later_still_reaches_every_platform() -> None:
    """An unrecognized floor key is carried everywhere rather than silently dropped."""
    floors = SystemFloors(declared={"quantum": "1.0"})
    assert floors.on(_LINUX64) == floors.on(_OSX) == {"quantum": "1.0"}


def test_an_undeclared_platform_list_compiles_for_this_machine() -> None:
    """A zero-config manifest still compiles to a workspace pixi can install here."""
    matrix = PlatformMatrix.from_manifest(Manifest(workspace=Header(name="zero-config")))
    assert matrix.workspace == [current_platform()]
