# What this machine could and should update, and the one pass that updates it.
#
# Every system keeps its software current through its own managers (apt, dnf, brew, snap, fwupd
# for firmware) plus the tools these workspaces add on top (pixi's global tools, uv, the owner's
# dotfiles with the toolbox their mb.lock pins, this tool). `audit` asks each manager present what
# is pending, read-only, and says what to run; `upgrade` runs the lot in order, the way the owner
# used to type `apt update && apt full-upgrade && apt autoremove` by hand. A host answers with its
# own copy of this tool, so the same code judges Linux and macOS.

import platform
import re
import shlex
import shutil
import time
from collections.abc import Callable, Sequence
from pathlib import Path

from ..core.constants.units import BYTES_PER_GB, SECONDS_PER_DAY
from ..core.execution import host
from ..core.project import Project
from ..core.section import Section
from ..enums import Verdict
from ..log import logger
from . import staleness
from .managers import Manager, ask, attached, lines, present, unknown

# Below this share of the home filesystem free, the audit warns.
_FREE_FLOOR = 0.10

# How old apt's package lists may be before their answer is stale.
_LISTS_DAYS = 7


def audit(managers: Sequence[Manager] | None = None) -> list[Section]:
    """What this machine could and should update, one row each, read-only.

    A row that warns names the command that fixes it; `host upgrade` runs every such command
    except firmware, which is the owner's to schedule. A manager that does not answer within its
    `timeout` warns as unknown instead, naming the command to run by hand.

    managers: the package managers to ask when their program is here; the known ones by default.
    """
    rows: list[Section | None] = [_system()]
    for manager in present(managers):
        said = ask(manager.pending, timeout=manager.timeout)
        if said is None:
            rows.append(unknown(manager.name, manager.pending, timeout=manager.timeout))
        else:
            pending = manager.count(said)
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


def _upgrade_steps() -> list[str]:
    """Every command `upgrade` runs here, in order.

    The system's managers come first, then the toolbox, the dotfiles and this tool.
    """
    steps = [step for manager in present() if not manager.audit_only for step in manager.steps]
    if shutil.which("pixi"):
        steps.append("pixi global update")
    # chezmoi's own clone is pulled; a source the owner works in (the center's checkout) is only
    # applied, since a pull there would rebase their unpushed work.
    if (source := Path.home() / ".local" / "share" / "chezmoi").is_dir():
        steps.append(f"pixi exec chezmoi --source {shlex.quote(source.as_posix())} update --force")
    elif shutil.which("chezmoi"):
        steps.append("chezmoi apply --force")
    steps.append(f"{Project().name} host update")
    return steps


def upgrade(*, dry_run: bool = False, run: Callable[[Sequence[str]], int] | None = None) -> int:
    """Run every upgrade step here, printing each first; stops at the first that fails.

    dry_run: only print the steps.
    run: runs one argv with the terminal attached, answering its exit status.
    """
    run = run or attached
    for step in _upgrade_steps():
        logger.result(f"$ {step}")
        if dry_run:
            continue
        status = run(shlex.split(step))
        if status:
            logger.result(f"stopped: `{step}` exited {status}")
            return status
    return 0


def _system() -> Section:
    """The operating system and kernel, for the record."""
    name = platform.system()
    release = Path("/etc/os-release")
    if name == "Linux" and release.is_file():
        found = re.search(r'^PRETTY_NAME="?([^"\n]+)', release.read_text(encoding="utf-8"), re.M)
        name = found.group(1) if found else name
    elif name == "Darwin":
        name = f"macOS {platform.mac_ver()[0]}, kernel {platform.release()}"
    if host.is_linux():
        name += f", kernel {platform.release()}"
    return Section(section="system", verdict=Verdict.PASS, detail=f"{name}, {platform.machine()}")


def _apt_lists() -> Section | None:
    """How fresh apt's package lists are, since `apt list --upgradable` only reads them."""
    stamp = Path("/var/cache/apt/pkgcache.bin")
    if not stamp.exists():
        return None
    days = (time.time() - stamp.stat().st_mtime) / SECONDS_PER_DAY
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
        detail=f"{usage.free / BYTES_PER_GB:.0f} GB free ({share:.0%})",
        fix="pixi clean cache; uv cache prune; docker system prune" if share < _FREE_FLOOR else "",
    )


def _snapshot() -> Section:
    """Whether this tool's installed snapshot answers for its source."""
    found = staleness.check()
    return Section(
        section=Project().name,
        verdict=Verdict.WARN if found.stale else Verdict.PASS,
        detail=found.detail,
        fix=f"{Project().name} host update" if found.stale else "",
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
    said = ask(argv)
    if said is None:
        return unknown("dotfiles", argv)
    drift = lines(said)
    return Section(
        section="dotfiles",
        verdict=Verdict.WARN if drift else Verdict.PASS,
        detail=f"{drift} file(s) differ from the dotfiles" if drift else "applied",
        fix="chezmoi apply" if drift else "",
    )
