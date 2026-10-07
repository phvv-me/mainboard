# What this machine could and should update, and the one pass that updates it.
#
# Every system keeps its software current through its own managers (apt, dnf, brew, snap, fwupd
# for firmware) plus the tools these workspaces add on top (the pixi global toolbox, uv, the
# owner's dotfiles, this tool). `audit` asks each manager present what is pending, read-only, and
# says what to run; `upgrade` runs the lot in order, the way the owner used to type `apt update &&
# apt full-upgrade && apt autoremove` by hand. A host answers with its own copy of this tool, so
# the same code judges Linux and macOS.

import json
import os
import platform
import re
import shlex
import shutil
import subprocess  # ruff:ignore[suspicious-subprocess-import]  reason=asks the machine's own package managers, fixed argv only since=2026-09-29
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from . import staleness
from .core.host import LINUX
from .core.project import Project
from .core.section import Section, Verdict

# How long one manager may take to say what is pending before it is skipped.
_ASK_SECONDS = 120

# Below this share of the home filesystem free, the audit warns.
_FREE_FLOOR = 0.10

# How old apt's package lists may be before their answer is stale.
_LISTS_DAYS = 7


def _lines(out: str) -> int:
    """How many non-blank lines a manager printed."""
    return sum(1 for line in out.splitlines() if line.strip())


def _apt(out: str) -> int:
    return sum("upgradable from" in line for line in out.splitlines())


def _snap(out: str) -> int:
    return 0 if "up to date" in out else max(_lines(out) - 1, 0)


def _fwupd(out: str) -> int:
    try:
        devices = json.loads(out).get("Devices", [])
    except json.JSONDecodeError, AttributeError:
        return 0
    return sum(1 for device in devices if device.get("Releases"))


@dataclass(frozen=True)
class Manager:
    """One system package manager: how to ask what is pending, and how to apply it.

    name: what the row calls it.
    program: the executable whose presence means the manager is here.
    pending: the read-only command listing what would change.
    count: how many updates `pending`'s output names.
    steps: the commands that apply them, in order; `sudo` where the manager needs root.
    audit_only: pending updates are reported but `upgrade` leaves them to the owner (firmware).
    """

    name: str
    program: str
    pending: tuple[str, ...]
    count: Callable[[str], int]
    steps: tuple[str, ...]
    audit_only: bool = False


MANAGERS = (
    Manager(
        "apt",
        "apt-get",
        ("apt", "list", "--upgradable"),
        _apt,
        (
            "sudo apt-get update",
            "sudo apt-get -y full-upgrade",
            "sudo apt-get -y autoremove --purge",
            "sudo apt-get -y autoclean",
        ),
    ),
    Manager(
        "dnf",
        "dnf",
        ("dnf", "-q", "check-update"),
        _lines,
        ("sudo dnf -y upgrade --refresh", "sudo dnf -y autoremove"),
    ),
    Manager(
        "brew",
        "brew",
        ("brew", "outdated", "--quiet"),
        _lines,
        ("brew update", "brew upgrade", "brew cleanup"),
    ),
    Manager("snap", "snap", ("snap", "refresh", "--list"), _snap, ("sudo snap refresh",)),
    Manager(
        "firmware",
        "fwupdmgr",
        ("fwupdmgr", "get-updates", "--json"),
        _fwupd,
        ("sudo fwupdmgr refresh --force", "sudo fwupdmgr update"),
        audit_only=True,
    ),
)


def pixi_home() -> Path:
    """The pixi home holding the global toolbox: the dotfiles' per-architecture one when it
    exists (Linux and macOS), else pixi's own."""
    arch = Path.home() / ".pixi" / platform.machine()
    return arch if (arch / "manifests").is_dir() else Path.home() / ".pixi"


def _ask(argv: Sequence[str]) -> str:
    """What a read-only command printed, empty when it could not answer in time."""
    try:
        done = subprocess.run(
            list(argv),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=_ASK_SECONDS,
            check=False,
            env={**os.environ, "PIXI_HOME": str(pixi_home())},
        )
    except OSError, subprocess.TimeoutExpired:
        return ""
    return done.stdout + done.stderr


def _present() -> list[Manager]:
    return [manager for manager in MANAGERS if shutil.which(manager.program)]


def audit() -> list[Section]:
    """What this machine could and should update, one row each, read-only.

    A row that warns names the command that fixes it; `host upgrade` runs every such command
    except firmware, which is the owner's to schedule.
    """
    rows: list[Section | None] = [_system()]
    for manager in _present():
        pending = manager.count(_ask(manager.pending))
        fix = " && ".join(manager.steps)
        rows.append(
            Section(
                section=manager.name,
                verdict=Verdict.WARN if pending else Verdict.PASS,
                detail=f"{pending} update(s) pending" if pending else "up to date",
                fix=fix if pending else "",
            )
        )
        if manager.name == "apt":
            rows.append(_apt_lists())
    rows += [_reboot(), _disk(), _snapshot(), _dotfiles()]
    return [row for row in rows if row is not None]


def upgrade_steps() -> list[str]:
    """Every command `upgrade` runs here, in order: the system's managers, then the toolbox,
    the dotfiles and this tool."""
    steps = [step for manager in _present() if not manager.audit_only for step in manager.steps]
    if shutil.which("pixi"):
        steps.append("pixi global update")
    # chezmoi's own clone is pulled; a source the owner works in (the center's checkout) is only
    # applied, since a pull there would rebase their unpushed work.
    if (source := Path.home() / ".local" / "share" / "chezmoi").is_dir():
        steps.append(f"pixi exec chezmoi --source {shlex.quote(source.as_posix())} update --force")
    elif shutil.which("chezmoi"):
        steps.append("chezmoi apply --force")
    steps.append(f"{Project().name} self update")
    return steps


def upgrade(*, dry_run: bool = False, run: Callable[[Sequence[str]], int] | None = None) -> int:
    """Run every upgrade step here, printing each first; stops at the first that fails.

    dry_run: only print the steps.
    run: runs one argv with the terminal attached, answering its exit status.
    """
    run = run or _attached
    for step in upgrade_steps():
        print(f"$ {step}", flush=True)
        if dry_run:
            continue
        status = run(shlex.split(step))
        if status:
            print(f"stopped: `{step}` exited {status}", flush=True)
            return status
    return 0


def _attached(argv: Sequence[str]) -> int:
    return subprocess.call(list(argv), env={**os.environ, "PIXI_HOME": str(pixi_home())})


def _system() -> Section:
    """The operating system and kernel, for the record."""
    name = platform.system()
    release = Path("/etc/os-release")
    if name == "Linux" and release.is_file():
        found = re.search(r'^PRETTY_NAME="?([^"\n]+)', release.read_text(encoding="utf-8"), re.M)
        name = found.group(1) if found else name
    elif name == "Darwin":
        name = f"macOS {platform.mac_ver()[0]}, kernel {platform.release()}"
    if LINUX:
        name += f", kernel {platform.release()}"
    return Section(section="system", verdict=Verdict.PASS, detail=f"{name}, {platform.machine()}")


def _apt_lists() -> Section | None:
    """How fresh apt's package lists are, since `apt list --upgradable` only reads them."""
    stamp = Path("/var/cache/apt/pkgcache.bin")
    if not stamp.exists():
        return None
    days = (time.time() - stamp.stat().st_mtime) / 86400
    stale = days > _LISTS_DAYS
    return Section(
        section="apt lists",
        verdict=Verdict.WARN if stale else Verdict.PASS,
        detail=f"last refreshed {days:.0f} day(s) ago",
        fix="sudo apt-get update" if stale else "",
    )


def _reboot() -> Section | None:
    """Whether an installed update waits for a reboot (Debian and Ubuntu say so in a file)."""
    flag = Path("/var/run/reboot-required")
    if not flag.parent.is_dir():
        return None
    if not flag.exists():
        return Section(section="reboot", verdict=Verdict.PASS, detail="none pending")
    packages = Path(f"{flag}.pkgs")
    held = packages.read_text(encoding="utf-8").split() if packages.is_file() else []
    detail = f"required by {', '.join(sorted(set(held))[:6])}" if held else "required"
    return Section(section="reboot", verdict=Verdict.WARN, detail=detail, fix="sudo reboot")


def _disk() -> Section:
    """Free space where the home lives."""
    usage = shutil.disk_usage(Path.home())
    share = usage.free / usage.total
    return Section(
        section="disk",
        verdict=Verdict.WARN if share < _FREE_FLOOR else Verdict.PASS,
        detail=f"{usage.free / 1e9:.0f} GB free ({share:.0%})",
        fix="pixi clean cache; uv cache prune; docker system prune" if share < _FREE_FLOOR else "",
    )


def _snapshot() -> Section:
    """Whether this tool's installed snapshot answers for its source."""
    found = staleness.check()
    return Section(
        section=Project().name,
        verdict=Verdict.WARN if found.stale else Verdict.PASS,
        detail=found.detail,
        fix=f"{Project().name} self update" if found.stale else "",
    )


def _dotfiles() -> Section | None:
    """Files the dotfiles would change here, drift that `chezmoi apply` settles."""
    source = Path.home() / ".local" / "share" / "chezmoi"
    if source.is_dir():
        argv = ["pixi", "exec", "chezmoi", "--source", source.as_posix(), "status"]
    elif shutil.which("chezmoi"):
        argv = ["chezmoi", "status"]
    else:
        return None
    drift = _lines(_ask(argv))
    return Section(
        section="dotfiles",
        verdict=Verdict.WARN if drift else Verdict.PASS,
        detail=f"{drift} file(s) differ from the dotfiles" if drift else "applied",
        fix="chezmoi apply" if drift else "",
    )
