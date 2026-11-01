"""The settler's timer and `serve`'s resident service in the systemd user manager."""

from pathlib import Path

import pytest

from mainboard import MissionError
from mainboard.core.services.base import Every
from mainboard.core.services.systemd import SystemdUnit, SystemdUser

from .fakes import Scripted

_ACTIVE = (0, "ActiveState=active\nLastTriggerUSec=n/a\n")
_LINGERING = (0, "Linger=yes\n")
_USER = ("systemctl", "--user")
_SHOW = (*_USER, "show")


@pytest.fixture
def timer(tmp_path: Path, manager: Scripted, tool: Path) -> SystemdUser:
    """The settler as a systemd timer, installed and active on a period of twenty minutes."""
    del tool
    installed = SystemdUser(tmp_path, units=tmp_path / "units", shell=manager)
    installed.install(Every.parse("20m"))
    return installed


def test_the_settler_timer_runs_its_pass_from_the_workspace_and_is_armed(
    tmp_path: Path, tool: Path
) -> None:
    shell = Scripted({_SHOW: _ACTIVE, ("loginctl",): _LINGERING})
    settler = SystemdUser(tmp_path, units=tmp_path / "units", shell=shell)
    settling = settler.install(Every.parse("20m"))
    service = settler.service.read_text(encoding="utf-8")
    assert f"WorkingDirectory={tmp_path}\nExecStart={tool} job list --json\n" in service
    assert "OnUnitActiveSec=20m\n" in settler.timer.read_text(encoding="utf-8")
    assert [*_USER, "enable", "--now", settler.timer.name] in shell.asked
    assert (settling.active, settling.every, settling.fix) == (True, "20m", "")


def test_a_settler_whose_user_does_not_linger_is_told_the_command_that_fixes_it(
    tmp_path: Path, tool: Path
) -> None:
    del tool
    shell = Scripted({_SHOW: _ACTIVE, ("loginctl",): (0, "Linger=no")})
    settling = SystemdUser(tmp_path, units=tmp_path / "units", shell=shell).install(
        Every.parse("1h")
    )
    assert settling.fix.startswith("loginctl enable-linger ")
    assert "a reboot stops it until this user lingers" in settling.detail


def test_removing_the_settler_disarms_the_timer_and_deletes_both_units(
    tmp_path: Path, tool: Path
) -> None:
    del tool
    shell = Scripted()
    settler = SystemdUser(tmp_path, units=tmp_path / "units", shell=shell)
    settler.install(Every.parse("20m"))
    settling = settler.remove()
    assert not settler.timer.exists() and not settler.service.exists()
    assert [*_USER, "disable", "--now", settler.timer.name] in shell.asked
    assert not settling.installed


def test_the_resident_service_restarts_always_and_runs_its_command_unexpanded(
    tmp_path: Path,
) -> None:
    shell = Scripted({_SHOW: _ACTIVE})
    unit = SystemdUnit(tmp_path, units=tmp_path / "units", shell=shell)
    command = ("/env/python", "-I", "/src/keeper.py", "/w/.mb/serve", "--", "pc", "50%", "$HOME")
    unit.install(command, stop_seconds=27, log=tmp_path / "state" / "unit.log")
    text = (tmp_path / "units" / unit.name).read_text(encoding="utf-8")
    assert "ExecStart=/env/python -I /src/keeper.py /w/.mb/serve -- pc 50%% '$$HOME'\n" in text
    assert all(
        line in text.splitlines()
        for line in ("Restart=always", "RestartSec=5", "TimeoutStopSec=27", "UMask=0077")
    )
    assert f"StandardError=append:{tmp_path / 'state' / 'unit.log'}" in text
    assert unit.installed() and unit.loaded()


def test_an_enable_systemd_refuses_is_a_refusal_naming_the_unit(tmp_path: Path) -> None:
    shell = Scripted({(*_USER, "enable"): (1, "Failed to connect to bus")})
    unit = SystemdUnit(tmp_path, units=tmp_path / "units", shell=shell)
    with pytest.raises(MissionError, match=rf"{unit.name}: Failed to connect to bus"):
        unit.install(("/bin/true",), stop_seconds=10, log=tmp_path / "unit.log")
    assert shell.asked[-1] == [*_USER, "enable", "--now", unit.name]


def test_the_resident_service_counts_as_loaded_only_while_systemd_reports_it_active(
    tmp_path: Path,
) -> None:
    shell = Scripted({_SHOW: (0, "ActiveState=activating\n")})
    unit = SystemdUnit(tmp_path, units=tmp_path / "units", shell=shell)
    unit.install(("/bin/true",), stop_seconds=10, log=tmp_path / "unit.log")
    unit.remove()
    assert not unit.loaded() and not unit.installed()
    assert [*_USER, "disable", "--now", unit.name] in shell.asked


def test_stopping_the_systemd_settler_stops_both_units_and_keeps_their_files(
    timer: SystemdUser, manager: Scripted, arming: str
) -> None:
    said = timer.stop()

    assert [*_USER, "stop", timer.timer.name, timer.service.name] in manager.asked
    assert timer.timer.is_file() and timer.service.is_file()
    assert said.startswith(f"stopped {timer.timer.name}: no pass runs until `{arming}`")
    assert timer.state().fix == arming


def test_stopping_a_systemd_settler_twice_says_the_second_found_it_stopped(
    timer: SystemdUser, manager: Scripted
) -> None:
    timer.stop()
    said = timer.stop()

    assert said == f"{timer.timer.name} is already stopped; nothing to stop"
    assert manager.user_verbs().count("stop") == 1


def test_starting_a_stopped_systemd_settler_enables_its_timer_again(
    timer: SystemdUser, manager: Scripted
) -> None:
    timer.stop()
    said = timer.start()

    assert manager.active == {timer.timer.name}
    assert said == f"started {timer.timer.name}: one pass every 20m, the first one period from now"
    assert timer.state().active


def test_starting_a_systemd_settler_that_is_active_says_so_and_enables_nothing(
    timer: SystemdUser, manager: Scripted
) -> None:
    said = timer.start()

    assert said == f"{timer.timer.name} is already active, one pass every 20m; nothing to start"
    assert manager.user_verbs().count("enable") == 1


def test_a_systemd_settler_nothing_installed_cannot_start_and_has_nothing_to_stop(
    manager: Scripted, tmp_path: Path
) -> None:
    absent = SystemdUser(tmp_path, units=tmp_path / "units", shell=manager)

    with pytest.raises(MissionError, match="`mb job list --every 20m` installs one"):
        absent.start()

    assert absent.stop() == f"{absent.timer.name} is already stopped; nothing to stop"
    assert {"enable", "stop"}.isdisjoint(manager.user_verbs())
