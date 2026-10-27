import platform
from pathlib import Path

import pytest

from mainboard.upkeep.machine import _pixi_home

_ARCH = f".pixi/{platform.machine()}"


@pytest.mark.parametrize(
    ("present", "home"),
    [
        pytest.param(f"{_ARCH}/manifests", _ARCH, id="a-global-manifest-in-the-architecture-home"),
        pytest.param(f"{_ARCH}/bin", _ARCH, id="pixi-installed-there-and-no-global-tool-yet"),
        pytest.param(".pixi/bin", ".pixi", id="pixi-in-its-own-home-only"),
        pytest.param("unrelated", ".pixi", id="no-pixi-home-at-all"),
    ],
)
def test_the_pixi_home_is_the_architecture_one_wherever_pixi_lives_in_it(
    present: str, home: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / present).mkdir(parents=True)
    monkeypatch.setenv("HOME", str(tmp_path))

    assert _pixi_home() == tmp_path / home
