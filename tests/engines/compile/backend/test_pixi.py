import platform
from collections.abc import Mapping
from pathlib import Path

import pytest

from mainboard import MissionError
from mainboard.engines.compile.backend import CommandResult, Pixi


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


@pytest.mark.parametrize(
    ("env", "resolve", "command"),
    [
        pytest.param("default", False, "mainboard install", id="default-locked"),
        pytest.param("training", True, "mainboard install training --resolve", id="named-resolve"),
    ],
)
def test_windows_home_storage_failure_names_the_outside_sandbox_provisioning_command(
    pixi: Pixi,
    monkeypatch: pytest.MonkeyPatch,
    env: str,
    *,
    resolve: bool,
    command: str,
) -> None:
    """A restricted profile failure explains where and how to retry the same provision."""
    result = CommandResult(
        1,
        "",
        "Error: FileStorageError: Could not determine the home directory",
    )
    monkeypatch.setattr(platform, "system", lambda: "Windows")
    monkeypatch.setattr(Pixi, "environment_result", lambda *args, **kwargs: result)

    with pytest.raises(MissionError, match="outside the restricted application sandbox") as caught:
        pixi.install(env, resolve=resolve)

    assert f"`{command}`" in str(caught.value)


@pytest.mark.parametrize(
    ("operating_system", "stderr"),
    [
        pytest.param("Windows", "network request timed out", id="unrelated-windows-failure"),
        pytest.param(
            "Linux",
            "FileStorageError: Could not determine the home directory",
            id="same-text-on-another-platform",
        ),
    ],
)
def test_other_pixi_install_failures_keep_the_generic_diagnostic(
    pixi: Pixi,
    monkeypatch: pytest.MonkeyPatch,
    operating_system: str,
    stderr: str,
) -> None:
    """Only the known Windows profile signature is attributed to an application sandbox."""
    result = CommandResult(1, "", stderr)
    monkeypatch.setattr(platform, "system", lambda: operating_system)
    monkeypatch.setattr(Pixi, "environment_result", lambda *args, **kwargs: result)

    with pytest.raises(MissionError, match=r"`pixi install` failed"):
        pixi.install("default")
