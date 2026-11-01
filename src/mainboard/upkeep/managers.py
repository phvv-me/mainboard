# The system package managers a machine may have (apt, dnf, brew, snap, fwupd for firmware): how
# to ask each what is pending without changing anything, how to apply it, and the two ways the
# audit and the upgrade run a command, both with pixi's home set the way the dotfiles lay it out.

import json
import os
import platform
import shlex
import shutil
import subprocess  # ruff:ignore[suspicious-subprocess-import]  reason=asks the machine's own package managers, fixed argv only since=2026-09-29
from collections.abc import Callable, Sequence
from pathlib import Path

from patos import FrozenModel

from ..core import Section, Verdict, captured
from ..core.constants.units import ASK_SECONDS


def lines(out: str) -> int:
    """How many non-blank lines a manager printed."""
    return sum(1 for line in out.splitlines() if line.strip())


def _apt(out: str) -> int:
    return sum("upgradable from" in line for line in out.splitlines())


def _snap(out: str) -> int:
    return 0 if "up to date" in out else max(lines(out) - 1, 0)


def _fwupd(out: str) -> int:
    try:
        devices = json.loads(out).get("Devices", [])
    except json.JSONDecodeError, AttributeError:
        return 0
    return sum(1 for device in devices if device.get("Releases"))


class Manager(FrozenModel):
    """One system package manager: how to ask what is pending, and how to apply it.

    name: what the row calls it.
    program: the executable whose presence means the manager is here.
    pending: the read-only command listing what would change.
    count: how many updates `pending`'s output names.
    steps: the commands that apply them, in order; `sudo` where the manager needs root.
    audit_only: pending updates are reported but `upgrade` leaves them to the owner (firmware).
    timeout: seconds `pending` may take before the audit calls the manager unknown.
    """

    name: str
    program: str
    pending: tuple[str, ...]
    count: Callable[[str], int]
    steps: tuple[str, ...]
    audit_only: bool = False
    timeout: float = ASK_SECONDS


_MANAGERS = (
    Manager(
        name="apt",
        program="apt-get",
        pending=("apt", "list", "--upgradable"),
        count=_apt,
        steps=(
            "sudo apt-get update",
            "sudo apt-get -y full-upgrade",
            "sudo apt-get -y autoremove --purge",
            "sudo apt-get -y autoclean",
        ),
    ),
    Manager(
        name="dnf",
        program="dnf",
        pending=("dnf", "-q", "check-update"),
        count=lines,
        steps=("sudo dnf -y upgrade --refresh", "sudo dnf -y autoremove"),
    ),
    Manager(
        name="brew",
        program="brew",
        pending=("brew", "outdated", "--quiet"),
        count=lines,
        steps=("brew update", "brew upgrade", "brew cleanup"),
    ),
    Manager(
        name="snap",
        program="snap",
        pending=("snap", "refresh", "--list"),
        count=_snap,
        steps=("sudo snap refresh",),
    ),
    Manager(
        name="firmware",
        program="fwupdmgr",
        pending=("fwupdmgr", "get-updates", "--json"),
        count=_fwupd,
        steps=("sudo fwupdmgr refresh --force", "sudo fwupdmgr update"),
        audit_only=True,
    ),
)


def _pixi_home() -> Path:
    """The pixi home pixi's global tools live in.

    That is the per-architecture one the dotfiles and this tool install pixi into when it exists
    (Linux and macOS), else pixi's own. It exists from the day pixi is there, not from the day a
    global tool is: the toolbox is mb.toml's, so a host may never have had a global manifest.
    """
    arch = Path.home() / ".pixi" / platform.machine()
    return arch if arch.is_dir() else Path.home() / ".pixi"


def ask(argv: Sequence[str], *, timeout: float = ASK_SECONDS) -> str | None:
    """What a read-only command printed, None when it could not answer in `timeout` seconds."""
    try:
        done = captured(argv, timeout=timeout, env={**os.environ, "PIXI_HOME": str(_pixi_home())})
    except OSError, subprocess.TimeoutExpired:
        return None
    return done.stdout + done.stderr


def attached(argv: Sequence[str]) -> int:
    """The exit status of a command run with the terminal attached, so `sudo` can ask."""
    return subprocess.call(list(argv), env={**os.environ, "PIXI_HOME": str(_pixi_home())})


def present(managers: Sequence[Manager] | None = None) -> list[Manager]:
    """The managers this machine has.

    managers: the candidates; the five systems this tool knows (apt, dnf, brew, snap, firmware)
        when none are given.
    """
    candidates = _MANAGERS if managers is None else managers
    return [manager for manager in candidates if shutil.which(manager.program)]


def unknown(name: str, argv: Sequence[str], *, timeout: float = ASK_SECONDS) -> Section:
    """The row for a command that gave no answer in time; its fix is to run it and see why."""
    return Section(
        section=name,
        verdict=Verdict.WARN,
        detail=f"unknown, no answer in {timeout:g} s",
        fix=shlex.join(argv),
    )
