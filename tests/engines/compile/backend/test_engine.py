from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from plumbum import local
from plumbum.commands.processes import CommandNotFound

from mainboard import MissionError
from mainboard.engines.compile.backend import PixiEngine

if TYPE_CHECKING:
    from pytest_subprocess import FakeProcess

_SH = str(local.which("sh"))


class _FakeLocalMissingPixi:
    """A plumbum `local` stand-in where `pixi` is off PATH but any other name resolves."""

    def __getitem__(self, key: str) -> str:
        if key == "pixi":
            raise CommandNotFound("pixi", [])
        return key


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

    binary = isolated_pixi_home / "bin" / "pixi"
    binary.parent.mkdir(parents=True)
    binary.touch()
    monkeypatch.setattr("mainboard.engines.compile.backend.engine.local", _FakeLocalMissingPixi())
    assert PixiEngine().command == str(binary)


def test_installed_binary_bootstraps_only_when_missing(
    monkeypatch: pytest.MonkeyPatch, isolated_pixi_home: Path
) -> None:
    binary = isolated_pixi_home / "bin" / "pixi"
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
    fp.register([_SH, "-c", fp.any()], returncode=0)
    PixiEngine().bootstrap()
    assert any("pixi.sh/install.sh" in str(arg) for call in fp.calls for arg in call)

    fp.register([_SH, "-c", fp.any()], returncode=1)
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
    monkeypatch.setenv("SHELL", shell)
    fp.register([_SH, "-c", fp.any()], returncode=0)
    PixiEngine().bootstrap()
    printed = capsys.readouterr().err
    assert "installing pixi engine" in printed
    assert ("adds a PATH line" in printed) == bool(notice)
    assert notice in printed
