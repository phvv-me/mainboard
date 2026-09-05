import platform
from collections.abc import Mapping
from pathlib import Path

import pytest
from plumbum import CommandNotFound

from mainboard import MissionError
from mainboard.engines.compile.backend import (
    PIXI_VERSION,
    CommandResult,
    Pixi,
    PixiEngine,
    engine,
)


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


class _Answering:
    """A resolved command answering one version string to any flag it is handed."""

    def __init__(self, version: str) -> None:
        self.version = version

    def __getitem__(self, flag: str):
        return lambda: f"pixi {self.version}\n"


class _Machine:
    """A `plumbum.local` that resolves exactly the binaries it was told about."""

    def __init__(self, known: Mapping[str, str]) -> None:
        self.known = known

    def __getitem__(self, name: str):
        if name not in self.known:
            raise CommandNotFound(name, [])
        return _Answering(self.known[name])


@pytest.mark.parametrize(
    ("resolves", "expected"),
    [
        pytest.param("pixi", PIXI_VERSION, id="on-the-path"),
        pytest.param("", "0.79.0", id="only-in-pixi-home"),
        pytest.param(None, "", id="nowhere-at-all"),
    ],
)
def test_version_reads_the_pixi_this_machine_runs_and_never_installs_one(
    resolves: str | None,
    expected: str,
    pixi: Pixi,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`facts`, `doctor` and every host alignment ask this before deciding anything.

    So it must never bootstrap: asking a machine what it runs cannot be what changes what it
    runs, and the old spelling went through the engine's resolved command, which installs pixi
    on a machine that has none. The empty name is pixi's own home, resolved here rather than at
    collection because `PIXI_HOME` is read per call.
    """
    named = str(PixiEngine.binary_path()) if resolves == "" else resolves
    monkeypatch.setattr(engine, "local", _Machine({named: expected} if named else {}))
    assert pixi.version() == expected
    assert PixiEngine().aligned() == (expected == PIXI_VERSION)
