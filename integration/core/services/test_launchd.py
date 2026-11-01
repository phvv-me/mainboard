"""The settler's and `serve`'s launchd agents: written, loaded, stopped and started."""

import os
import plistlib
from pathlib import Path

import pytest

from mainboard import MissionError
from mainboard.core.services.base import Every
from mainboard.core.services.launchd import LaunchdAgent, LaunchdUnit, Plist

from .fakes import Scripted

_GUI = f"gui/{os.getuid()}"
_LAUNCHCTL = "launchctl"
_BOOTSTRAP = "bootstrap"
_BOOTOUT = "bootout"
_KEEPER = ("/env/bin/python", "-I", "/src/mainboard/serve/keeper.py", "/w/.mb/serve", "--", "pc")


def _plist(agents: Path, label: str) -> Plist:
    return plistlib.loads((agents / f"{label}.plist").read_bytes())


@pytest.fixture
def agent(tmp_path: Path, manager: Scripted, tool: Path) -> LaunchdAgent:
    """The settler as a launchd agent, installed and loaded on a period of twenty minutes."""
    del tool
    installed = LaunchdAgent(tmp_path, agents=tmp_path / "agents", shell=manager)
    installed.install(Every.parse("20m"))
    return installed


def test_the_settler_runs_its_pass_over_ssh_from_the_workspace_on_its_period(
    tmp_path: Path, tool: Path
) -> None:
    shell = Scripted()
    agent = LaunchdAgent(tmp_path, agents=tmp_path / "agents", shell=shell)
    settling = agent.install(Every.parse("20m"))
    written = _plist(tmp_path / "agents", agent.label)
    assert written["ProgramArguments"] == [
        *("/usr/bin/ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "localhost"),
        f"cd {tmp_path} && {tool} job list --json",
    ]
    assert (written["StartInterval"], written["RunAtLoad"]) == (1200, True)
    assert (settling.installed, settling.active, settling.every, settling.root) == (
        True,
        True,
        "20m",
        str(tmp_path),
    )
    assert [verb for verb in shell.verbs(_LAUNCHCTL) if verb != "print"] == [
        "bootout",
        _BOOTSTRAP,
    ]


def test_the_resident_agent_logs_in_with_a_terminal_and_launchd_keeps_it_alive(
    tmp_path: Path,
) -> None:
    shell = Scripted()
    unit = LaunchdUnit(tmp_path, agents=tmp_path / "agents", shell=shell)
    unit.install(_KEEPER, stop_seconds=27, log=tmp_path / "state" / "unit.log")
    written = _plist(tmp_path / "agents", unit.name)
    assert written["ProgramArguments"] == [
        *("/usr/bin/ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "-tt", "localhost"),
        f"cd {tmp_path} && exec " + " ".join(_KEEPER),
    ]
    assert {key: written[key] for key in ("RunAtLoad", "KeepAlive", "ExitTimeOut")} == {
        "RunAtLoad": True,
        "KeepAlive": True,
        "ExitTimeOut": 27,
    }
    assert (written["StandardOutPath"], written["EnvironmentVariables"]) == (
        os.devnull,
        {"PATH": "/usr/bin:/bin:/usr/sbin:/sbin"},
    )
    assert [_LAUNCHCTL, _BOOTSTRAP, _GUI, str(tmp_path / "agents" / f"{unit.name}.plist")] in (
        shell.asked
    )
    assert unit.installed() and unit.loaded()


def test_an_install_refuses_when_ssh_localhost_will_not_log_in(tmp_path: Path) -> None:
    shell = Scripted({("/usr/bin/ssh",): (255, "Permission denied (publickey).")})
    unit = LaunchdUnit(tmp_path, agents=tmp_path / "agents", shell=shell)
    with pytest.raises(MissionError, match=r"Permission denied \(publickey\)"):
        unit.install(_KEEPER, stop_seconds=10, log=tmp_path / "unit.log")
    assert not (tmp_path / "agents").exists()
    assert _BOOTSTRAP not in shell.verbs(_LAUNCHCTL)


def test_a_bootstrap_launchd_refuses_is_a_refusal_naming_the_plist(tmp_path: Path) -> None:
    shell = Scripted({(_LAUNCHCTL, _BOOTSTRAP): (5, "Bootstrap failed: 5: Input/output error")})
    unit = LaunchdUnit(tmp_path, agents=tmp_path / "agents", shell=shell)
    with pytest.raises(MissionError, match=rf"{unit.name}\.plist: Bootstrap failed: 5"):
        unit.install(_KEEPER, stop_seconds=10, log=tmp_path / "unit.log")
    assert shell.asked[-1][:2] == [_LAUNCHCTL, _BOOTSTRAP]


def test_without_a_console_login_an_agent_goes_to_the_background_domain(tmp_path: Path) -> None:
    shell = Scripted({(_LAUNCHCTL, "print", _GUI): (113, "Could not find domain")})
    unit = LaunchdUnit(tmp_path, agents=tmp_path / "agents", shell=shell)
    unit.install(_KEEPER, stop_seconds=10, log=tmp_path / "unit.log")
    bootstrap = next(said for said in shell.asked if said[:2] == [_LAUNCHCTL, _BOOTSTRAP])
    assert bootstrap[2] == f"user/{os.getuid()}"


def test_removing_the_resident_agent_unloads_it_and_deletes_its_plist(tmp_path: Path) -> None:
    shell = Scripted()
    unit = LaunchdUnit(tmp_path, agents=tmp_path / "agents", shell=shell)
    unit.install(_KEEPER, stop_seconds=10, log=tmp_path / "unit.log")
    unit.remove()
    assert not unit.installed()
    assert shell.asked[-1] == [_LAUNCHCTL, "bootout", f"{_GUI}/{unit.name}"]


def test_a_workspace_names_its_settler_and_its_resident_agent_with_one_stamp(
    tmp_path: Path,
) -> None:
    agent, unit = LaunchdAgent(tmp_path), LaunchdUnit(tmp_path)
    assert agent.label.removeprefix("mainboard.monitor.") == unit.name.removeprefix(
        "mainboard.serve."
    )
    assert LaunchdUnit(tmp_path / "other").name != unit.name


def test_a_settler_launchd_does_not_hold_names_the_verb_that_loads_it(
    tmp_path: Path, tool: Path, arming: str
) -> None:
    del tool
    shell = Scripted()
    agent = LaunchdAgent(tmp_path, agents=tmp_path / "agents", shell=shell)
    shell.answers[(_LAUNCHCTL, "print", f"{_GUI}/{agent.label}")] = (113, "Could not find it")
    settling = agent.install(Every.parse("1h"))
    assert (settling.active, settling.every) == (False, "1h")
    assert settling.fix == arming


def test_stopping_the_launchd_settler_unloads_it_and_keeps_its_plist(
    agent: LaunchdAgent, manager: Scripted, arming: str
) -> None:
    said = agent.stop()

    assert manager.loaded == set()
    assert agent.plist.is_file()
    assert said.startswith(f"stopped {agent.label}: no pass runs until `{arming}`")
    assert agent.state().fix == arming


def test_stopping_a_launchd_settler_twice_says_the_second_found_it_unloaded(
    agent: LaunchdAgent, manager: Scripted
) -> None:
    agent.stop()
    said = agent.stop()

    assert said == f"{agent.label} is already unloaded; nothing to stop"
    assert manager.verbs(_LAUNCHCTL).count(_BOOTOUT) == 2


def test_starting_a_stopped_launchd_settler_loads_the_plist_it_kept(
    agent: LaunchdAgent, manager: Scripted
) -> None:
    agent.stop()
    said = agent.start()

    assert manager.loaded == {agent.label}
    assert said == f"started {agent.label}: a pass runs now, then one every 20m"
    assert agent.state().active


def test_starting_a_launchd_settler_that_is_loaded_says_so_and_loads_nothing(
    agent: LaunchdAgent, manager: Scripted
) -> None:
    said = agent.start()

    assert said == f"{agent.label} is already loaded, one pass every 20m; nothing to start"
    assert manager.verbs(_LAUNCHCTL).count(_BOOTSTRAP) == 1


def test_a_launchd_settler_nothing_installed_cannot_start_and_has_nothing_to_stop(
    manager: Scripted, tmp_path: Path
) -> None:
    absent = LaunchdAgent(tmp_path, agents=tmp_path / "agents", shell=manager)

    with pytest.raises(MissionError, match="`mb job list --every 20m` installs one"):
        absent.start()

    assert absent.stop() == f"{absent.label} is already unloaded; nothing to stop"
    assert {_BOOTSTRAP, _BOOTOUT}.isdisjoint(manager.verbs(_LAUNCHCTL))


def test_a_bootout_launchd_refuses_is_a_refusal_naming_the_agent(
    agent: LaunchdAgent, manager: Scripted
) -> None:
    manager.answers[(_LAUNCHCTL, _BOOTOUT)] = (1, "Boot-out failed: 1: Operation not permitted")

    with pytest.raises(MissionError, match=rf"unload {agent.label}: Boot-out failed: 1"):
        agent.stop()

    assert manager.loaded == {agent.label}
