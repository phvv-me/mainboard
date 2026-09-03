import platform
from collections.abc import Mapping
from pathlib import Path

import pytest

from mainboard.engines.compile.backend import Pixi


def manifest_with_floors(pixi: Pixi) -> None:
    descriptor = '{name = "linux-aarch64-system", platform = "linux-aarch64", cuda = "13.0"}'
    pixi.manifest.write_text(
        f'[workspace]\nplatforms = ["linux-64", {descriptor}]\n', encoding="utf-8"
    )


def test_floor_overrides_answer_empty_without_a_generated_manifest(pixi: Pixi) -> None:
    assert Pixi._floor_overrides(pixi.manifest) == {}


def test_floor_overrides_map_descriptor_floors_to_conda_override_vars(
    pixi: Pixi, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("CONDA_OVERRIDE_CUDA", raising=False)
    manifest_with_floors(pixi)
    assert Pixi._floor_overrides(pixi.manifest) == {"CONDA_OVERRIDE_CUDA": "13.0"}


def test_floor_overrides_leave_a_callers_own_export_standing(
    pixi: Pixi, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CONDA_OVERRIDE_CUDA", "12.4")
    manifest_with_floors(pixi)
    assert Pixi._floor_overrides(pixi.manifest) == {}


def test_command_vouches_declared_floors_through_its_environment(
    pixi: Pixi, tool_paths: Mapping[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("CONDA_OVERRIDE_CUDA", raising=False)
    expected = {"HOME": str(Path.home())} if platform.system() == "Windows" else {}
    assert dict(pixi.command.env or {}) == expected
    manifest_with_floors(pixi)
    expected |= {"CONDA_OVERRIDE_CUDA": "13.0"}
    assert dict(pixi.command.env or {}) == expected
