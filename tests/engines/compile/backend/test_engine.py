import os
import platform
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from plumbum import local
from plumbum.commands.base import BaseCommand
from plumbum.commands.processes import CommandNotFound

from mainboard import MissionError
from mainboard.engines.compile.backend import PixiEngine

if TYPE_CHECKING:
    from pytest_subprocess import FakeProcess


class _FakeLocalMissingPixi:
    """A plumbum `local` stand-in where `pixi` is off PATH but any other name resolves."""

    def __getitem__(self, key: str) -> BaseCommand:
        if key == "pixi":
            raise CommandNotFound("pixi", [])
        return local[key]


def test_home_prefers_pixi_home_and_falls_back_to_the_users_own_directory(
    monkeypatch: pytest.MonkeyPatch, isolated_pixi_home: Path
) -> None:
    assert PixiEngine.home() == isolated_pixi_home
    monkeypatch.delenv("PIXI_HOME")
    assert PixiEngine.home() == Path.home() / ".pixi"


def test_command_prefers_pixi_on_path_and_falls_back_to_pixi_home(
    monkeypatch: pytest.MonkeyPatch, tool_paths: Mapping[str, str], isolated_pixi_home: Path
) -> None:
    """A non-login remote shell can drop `PIXI_HOME/bin` from PATH without pixi being absent."""
    assert str(PixiEngine().command) == tool_paths["pixi"]

    binary = PixiEngine.binary_path()
    binary.parent.mkdir(parents=True)
    binary.touch()
    monkeypatch.setattr("mainboard.engines.compile.backend.engine.local", _FakeLocalMissingPixi())
    assert Path(PixiEngine().command.formulate()[0]) == binary


@pytest.mark.parametrize("inherited", [None, "C:/sandbox/profile"])
def test_windows_pixi_commands_supply_the_real_home_regardless_of_the_launchers_value(
    monkeypatch: pytest.MonkeyPatch,
    inherited: str | None,
) -> None:
    """Pixi receives the actual profile without changing the calling process environment."""
    monkeypatch.setattr("platform.system", lambda: "Windows")
    if inherited is None:
        monkeypatch.delenv("HOME", raising=False)
    else:
        monkeypatch.setenv("HOME", inherited)
    command = PixiEngine().command

    assert command.env["HOME"] == str(Path.home())
    assert os.environ.get("HOME") == inherited


def test_installed_binary_bootstraps_only_when_missing(
    monkeypatch: pytest.MonkeyPatch, isolated_pixi_home: Path
) -> None:
    binary = PixiEngine.binary_path()
    calls: list[bool] = []
    monkeypatch.setattr(PixiEngine, "bootstrap", lambda self: calls.append(True))

    assert PixiEngine().installed_binary() == binary
    assert calls == [True]

    calls.clear()
    binary.parent.mkdir(parents=True)
    binary.touch()
    assert PixiEngine().installed_binary() == binary
    assert calls == []


def test_bootstrap_runs_the_official_installer_and_raises_when_it_fails(fp: FakeProcess) -> None:
    """`pip install mainboard` brings no pixi binary, so the engine installs one on first use."""
    command = PixiEngine.installer()
    fp.register(command.formulate(), returncode=0)
    PixiEngine().bootstrap()
    suffix = "install.ps1" if platform.system() == "Windows" else "install.sh"
    assert any(suffix in str(arg) for call in fp.calls for arg in call)

    fp.register(command.formulate(), returncode=1)
    with pytest.raises(MissionError, match="pixi installer failed"):
        PixiEngine().bootstrap()


@pytest.mark.parametrize(
    ("environment", "appended"),
    [
        pytest.param({"SHELL": "/bin/zsh"}, "~/.zshrc", id="zsh"),
        pytest.param({"SHELL": "/usr/bin/fish"}, "~/.config/fish/config.fish", id="fish"),
        pytest.param({"SHELL": "/bin/ksh"}, "", id="a-shell-the-installer-has-no-rule-for"),
        pytest.param({"SHELL": ""}, "", id="no-shell-named-at-all"),
        pytest.param(
            {"SHELL": "/bin/zsh", "PIXI_NO_PATH_UPDATE": "1"},
            "",
            id="the-suppression-variable-means-nothing-is-touched",
        ),
    ],
)
def test_the_appended_shell_file_matches_what_the_installer_would_edit(
    monkeypatch: pytest.MonkeyPatch, environment: Mapping[str, str], appended: str
) -> None:
    """The installer edits a personal startup file, so the engine can name which one."""
    monkeypatch.delenv("PIXI_NO_PATH_UPDATE", raising=False)
    monkeypatch.setattr("platform.system", lambda: "Linux")
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    assert PixiEngine.appended_shell_file() == appended


@pytest.mark.parametrize(
    ("shell", "notice"),
    [
        pytest.param("/bin/bash", "adds a PATH line to ~/.bashrc", id="a-shell-it-will-edit"),
        pytest.param("/bin/ksh", "", id="a-shell-it-leaves-alone"),
    ],
)
def test_bootstrap_names_the_shell_file_the_installer_appends_to(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    fp: FakeProcess,
    shell: str,
    notice: str,
) -> None:
    """One printed line, so nobody discovers the edit by finding it in their own dotfile."""
    monkeypatch.delenv("PIXI_NO_PATH_UPDATE", raising=False)
    monkeypatch.setattr("platform.system", lambda: "Linux")
    monkeypatch.setattr("shutil.which", lambda name: sys.executable if name == "sh" else None)
    monkeypatch.setenv("SHELL", shell)
    command = PixiEngine.installer()
    fp.register(command.formulate(), returncode=0)
    PixiEngine().bootstrap()
    printed = capsys.readouterr().err
    assert "installing pixi engine" in printed
    assert ("adds a PATH line" in printed) == bool(notice)
    assert notice in printed


def test_windows_installer_uses_powershell_and_edits_no_shell_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("platform.system", lambda: "Windows")
    monkeypatch.setattr(
        "shutil.which", lambda name: "C:/Windows/powershell.exe" if name == "powershell" else None
    )
    command = PixiEngine.installer().formulate()
    assert Path(command[0]) == Path("C:/Windows/powershell.exe")
    assert "install.ps1" in command[-1]
    assert PixiEngine.appended_shell_file() == ""
