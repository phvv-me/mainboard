from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from plumbum import local
from plumbum.commands.processes import CommandNotFound

from mainboard import MissionError
from mainboard.engines.compile.backend import PixiEngine

if TYPE_CHECKING:
    from collections.abc import Mapping

    from pytest_subprocess import FakeProcess


class _FakeLocalMissingPixi:
    """A plumbum `local` stand-in where `pixi` is off PATH but any other name resolves."""

    def __getitem__(self, key: str) -> str:
        if key == "pixi":
            raise CommandNotFound("pixi", [])
        return key


def test_home_reads_pixi_home_from_the_environment(tmp_path: Path) -> None:
    assert PixiEngine.home() == tmp_path / "pixi-home"


def test_home_falls_back_to_the_users_pixi_directory(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PIXI_HOME", raising=False)
    assert PixiEngine.home() == Path.home() / ".pixi"


def test_command_prefers_pixi_already_on_path(tool_paths: Mapping[str, str]) -> None:
    assert str(PixiEngine().command) == tool_paths["pixi"]


def test_command_falls_back_to_pixi_home_when_off_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("mainboard.engines.compile.backend.engine.local", _FakeLocalMissingPixi())
    binary = tmp_path / "pixi-home" / "bin" / "pixi"
    binary.parent.mkdir(parents=True)
    binary.touch()
    assert PixiEngine().command == str(binary)


def test_installed_binary_bootstraps_only_when_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    binary = tmp_path / "pixi-home" / "bin" / "pixi"
    calls: list[bool] = []
    monkeypatch.setattr(PixiEngine, "bootstrap", lambda self: calls.append(True))

    assert PixiEngine().installed_binary() == binary
    assert calls == [True]

    calls.clear()
    binary.parent.mkdir(parents=True)
    binary.touch()
    assert PixiEngine().installed_binary() == binary
    assert calls == []


def test_bootstrap_runs_the_official_installer(fp: FakeProcess) -> None:
    fp.register([str(local.which("sh")), "-c", fp.any()], returncode=0)
    PixiEngine().bootstrap()
    assert any("pixi.sh/install.sh" in str(arg) for call in fp.calls for arg in call)


def test_bootstrap_raises_when_the_installer_fails(fp: FakeProcess) -> None:
    fp.register([str(local.which("sh")), "-c", fp.any()], returncode=1)
    with pytest.raises(MissionError, match="pixi installer failed"):
        PixiEngine().bootstrap()
