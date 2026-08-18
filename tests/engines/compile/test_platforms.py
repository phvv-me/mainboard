from typing import TYPE_CHECKING

from mainboard import Manifest
from mainboard.core.host import current_platform
from mainboard.engines.compile.platforms import PlatformMatrix
from mainboard.manifest import Header

if TYPE_CHECKING:
    from collections.abc import Callable

_LINUX64 = "linux-64"


def test_no_system_floors_anywhere_leaves_platforms_bare(
    manifest_from: Callable[[str], Manifest],
) -> None:
    """With no floor declared, platforms stay plain strings and nothing gets routed."""
    manifest = manifest_from(
        """
        [workspace]
        name = "w"
        platforms = ["linux-64", "linux-aarch64"]
        """
    )
    matrix = PlatformMatrix.from_manifest(manifest)
    assert matrix.workspace == [_LINUX64, "linux-aarch64"]
    assert matrix.environments == {}
    assert matrix.default == []


def test_a_root_system_floor_routes_every_env(
    manifest_from: Callable[[str], Manifest],
) -> None:
    """A workspace-wide floor names every platform and routes every declared env onto it."""
    manifest = manifest_from(
        """
        [workspace]
        name = "w"
        platforms = ["linux-64"]
        [system]
        cuda = "13.0"
        [envs.serving]
        """
    )
    matrix = PlatformMatrix.from_manifest(manifest)
    assert matrix.workspace == [{"name": "linux-64-system", "platform": _LINUX64, "cuda": "13.0"}]
    assert matrix.default == ["linux-64-system"]
    assert matrix.environments == {"serving": ["linux-64-system"]}


def test_an_env_raising_its_own_floor_gets_a_named_variant(
    manifest_from: Callable[[str], Manifest],
) -> None:
    """An env with a higher floor than the workspace gets its own `<platform>-<env>` variant."""
    manifest = manifest_from(
        """
        [workspace]
        name = "w"
        platforms = ["linux-64"]
        [envs.serving]
        system = { cuda = "13.0" }
        """
    )
    matrix = PlatformMatrix.from_manifest(manifest)
    assert _LINUX64 in matrix.workspace
    assert {
        "name": "linux-64-serving",
        "platform": _LINUX64,
        "cuda": "13.0",
    } in matrix.workspace
    assert matrix.environments == {"serving": ["linux-64-serving"]}
    # An env-level floor alone still routes the workspace's own default onto the root platforms.
    assert matrix.default == [_LINUX64]


def test_an_env_narrowing_its_own_platform_list_is_named_even_without_a_floor(
    manifest_from: Callable[[str], Manifest],
) -> None:
    """Any floor anywhere forces every env to name the platform variants it actually runs on."""
    manifest = manifest_from(
        """
        [workspace]
        name = "w"
        platforms = ["linux-64", "linux-aarch64"]
        [envs.serving]
        system = { cuda = "13.0" }
        [envs.cpu_only]
        platforms = ["linux-aarch64"]
        """
    )
    matrix = PlatformMatrix.from_manifest(manifest)
    # `cpu_only` raises no floor of its own, so it is routed onto the bare root platform name.
    assert matrix.environments["cpu_only"] == ["linux-aarch64"]
    assert matrix.environments["serving"] == ["linux-64-serving", "linux-aarch64-serving"]


def test_descriptor_is_a_bare_string_without_any_system_floor() -> None:
    assert PlatformMatrix.descriptor(_LINUX64, platform=_LINUX64, system={}) == _LINUX64


def test_descriptor_is_a_named_table_with_a_system_floor() -> None:
    assert PlatformMatrix.descriptor(
        "linux-64-serving", platform=_LINUX64, system={"cuda": "13.0"}
    ) == {"name": "linux-64-serving", "platform": _LINUX64, "cuda": "13.0"}


def test_empty_platform_list_defaults_to_this_machine() -> None:
    matrix = PlatformMatrix.from_manifest(Manifest(workspace=Header(name="zero-config")))
    assert matrix.workspace == [current_platform()]
