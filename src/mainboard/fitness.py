# Whether one machine, as its census describes it, can serve this workspace. The census says what
# a machine is and nothing about the workspace; this reads that description against what the
# manifest and its lock ask for, and answers in the one row shape every report here prints. It is
# the single judge behind the findings `facts`, `compute`, `setup`, `center verify` and `center
# migrate` show, so a driver too old for the lock is the same sentence wherever it surfaces.

import re
from collections import Counter
from enum import StrEnum, auto
from typing import TYPE_CHECKING

from packaging.version import Version

from .core.errors import MissionError
from .core.host import current_platform
from .core.section import Section, Verdict
from .engines.compile.pixi_lock import packages
from .engines.compile.platforms import SystemFloors
from .engines.compile.provisioner import Provisioner
from .git.process import Git
from .workstation import DEVELOPER_MODE, install_command

if TYPE_CHECKING:
    from pathlib import Path

    from .manifest.schema.host import HostProfile
    from .manifest.schema.root import Manifest
    from .probe.system import System

# The CUDA a locked build was compiled against, as a PyPI wheel spells it (`+cu132`, `/cu132/`)
# and as conda pins it (`cuda-version-13.0-...`).
_WHEEL_CUDA = re.compile(r"[+/]cu(\d{2})(\d)\b")
_CONDA_CUDA = re.compile(r"/cuda-version-(\d+\.\d+)-")

# The oldest CUDA whose builds carry kernels for each compute capability generation, newest
# first. A card newer than every kernel a build carries has nothing to run, which is what a
# Blackwell card on a CUDA 12.4 torch looks like: it imports, then fails its first launch.
_KERNELS = ((Version("10.0"), Version("12.8")), (Version("8.9"), Version("11.8")))

# Scheduler kinds whose ssh endpoint is a login node, whose cards say nothing about a job's.
_LOGIN_NODES = frozenset({"pbs", "slurm"})

# The free space a machine needs where the workspace lives before a clone or an environment can
# land, in bytes: a center holds the whole tree and every environment, a target one mirror.
_DISK = {"center": 60 * 10**9, "target": 20 * 10**9}

# The tools a center cannot work without, each installable with one named command.
_CENTER_TOOLS = ("git", "git-lfs", "gh", "ssh")

# The NVIDIA driver download page, the one fix for a driver below the workspace's floor.
_DRIVERS = "install a newer NVIDIA driver (https://www.nvidia.com/Download/index.aspx)"

# The Windows switch that lets paths past 260 characters open, set once by an administrator.
_LONG_PATHS = (
    "reg add HKLM\\SYSTEM\\CurrentControlSet\\Control\\FileSystem /v LongPathsEnabled "
    "/t REG_DWORD /d 1 /f (as administrator)"
)

# How many colliding paths a detail line names before it only counts the rest.
_NAMED = 3


class Role(StrEnum):
    """What a machine is for: the center that holds the monorepo, or a target that runs jobs."""

    CENTER = auto()
    TARGET = auto()


class Fitness:
    """Judge one machine's census against this workspace's manifest and lock.

    root: the workspace root, where the lock and the tracked paths are read.
    manifest: the loaded workspace manifest.
    """

    def __init__(self, root: Path, manifest: Manifest) -> None:
        self.root = root
        self.manifest = manifest
        self.locks: dict[str, dict[str, list[str]] | None] = {}

    def judge(
        self, system: System, *, host: str = "local", role: Role = Role.TARGET
    ) -> list[Section]:
        """Every finding about `system` as the machine `host` names, one row per question.

        A machine no census ever described is one row saying so, since every other question
        would only repeat that absence. A center is asked about its tooling and its checkout on
        top of what any machine is asked about the environment it installs.

        system: the machine's census.
        host: the alias whose profile says which environment and card the machine serves.
        role: what the machine is for.
        """
        if not system.surveyed:
            return [
                Section(
                    section="census",
                    verdict=Verdict.WARN,
                    detail="no software census recorded for this machine",
                    fix=f"mainboard setup {host}" if host != "local" else "",
                )
            ]
        profile = self.manifest.profile(host)
        environment = profile.env
        found = [
            self.platform(system),
            self.lock(system, environment),
            *self.cards(system, profile, environment),
            self.disk(system, role),
        ]
        if role is Role.CENTER:
            found += [
                self.tools(system),
                self.sync(system),
                self.links(system),
                self.case(system),
                self.line_endings(system),
                self.shell(system),
            ]
        return found

    def platform(self, system: System) -> Section:
        """Whether the workspace declares this machine's platform at all."""
        declared = self.manifest.workspace.platforms or [current_platform()]
        if system.platform in declared:
            return Section(
                section="platform",
                verdict=Verdict.PASS,
                detail=f"{system.summary()} is {system.platform}, which the workspace declares",
            )
        return Section(
            section="platform",
            verdict=Verdict.FAIL,
            detail=f"{system.platform} is not among the declared platforms {declared}",
            fix=(
                f'add "{system.platform}" to [workspace] platforms, then '
                "mainboard install --resolve"
            ),
        )

    def lock(self, system: System, environment: str) -> Section:
        """Whether the lock this workspace solved holds builds for this machine's platform."""
        install = f"mainboard install {environment} --resolve"
        held = self._locked(environment)
        if held is None:
            return Section(
                section="lock",
                verdict=Verdict.WARN,
                detail=f"{environment}: nothing solved yet",
                fix=install,
            )
        builds = held.get(system.platform, [])
        if not builds:
            return Section(
                section="lock",
                verdict=Verdict.FAIL,
                detail=f"{environment}: the lock holds no {system.platform} builds",
                fix=install,
            )
        return Section(
            section="lock",
            verdict=Verdict.PASS,
            detail=f"{environment}: {len(builds)} {system.platform} builds locked",
        )

    def cards(self, system: System, profile: HostProfile, environment: str) -> list[Section]:
        """The driver, the locked CUDA builds and the card memory, against what jobs need.

        A scheduler's ssh endpoint is its login node, whose cards (usually none) say nothing
        about the compute node a job lands on, so such a host answers once, with that caveat.
        """
        if profile.kind in _LOGIN_NODES:
            return [
                Section(
                    section="cards",
                    verdict=Verdict.PASS,
                    detail=f"{profile.kind} login node; cards are a compute node's, not read here",
                )
            ]
        floors = SystemFloors(declared=self._floors(environment)).on(system.platform)
        return [
            self.driver(system, floors.get("cuda", "")),
            self.builds(system, environment),
            self.memory(system, profile),
        ]

    def driver(self, system: System, floor: str) -> Section:
        """Whether the NVIDIA driver supports the CUDA floor the workspace solved against."""
        if not floor:
            return Section(
                section="driver",
                verdict=Verdict.PASS,
                detail=f"no CUDA floor on {system.platform}",
            )
        found = system.driver_cuda
        if found is None:
            return Section(
                section="driver",
                verdict=Verdict.WARN,
                detail=f"no NVIDIA driver answered, and the workspace solves CUDA {floor} builds",
                fix=_DRIVERS,
            )
        if found < Version(floor):
            return Section(
                section="driver",
                verdict=Verdict.FAIL,
                detail=f"the driver supports CUDA {found}, below the workspace floor {floor}",
                fix=_DRIVERS,
            )
        return Section(
            section="driver",
            verdict=Verdict.PASS,
            detail=f"driver CUDA {found} meets the floor {floor}",
        )

    def builds(self, system: System, environment: str) -> Section:
        """Whether the locked CUDA builds run on this driver and carry kernels for this card."""
        built = self._built(system.platform, environment)
        if built is None or not system.gpus:
            return Section(
                section="cuda-builds",
                verdict=Verdict.PASS,
                detail="no CUDA builds locked for this card to run",
            )
        driver = system.driver_cuda
        if driver is not None and driver < built:
            return Section(
                section="cuda-builds",
                verdict=Verdict.FAIL,
                detail=f"builds locked for CUDA {built}, a driver supporting {driver}",
                fix=_DRIVERS,
            )
        capability = system.capability
        needed = next(
            (cuda for generation, cuda in _KERNELS if capability and capability >= generation),
            None,
        )
        if needed is not None and built < needed:
            return Section(
                section="cuda-builds",
                verdict=Verdict.FAIL,
                detail=(
                    f"compute capability {capability} needs CUDA {needed} builds, "
                    f"the lock holds CUDA {built}"
                ),
                fix=f"raise the CUDA of the locked builds, then mainboard install {environment} "
                "--resolve",
            )
        return Section(
            section="cuda-builds",
            verdict=Verdict.PASS,
            detail=f"CUDA {built} builds run on this driver and card",
        )

    def memory(self, system: System, profile: HostProfile) -> Section:
        """Whether the cards hold what the profile declares a job there needs."""
        need = profile.defaults.vram_gb
        cards = len(system.gpus)
        wanted = profile.defaults.gpus
        if wanted > cards:
            return Section(
                section="memory",
                verdict=Verdict.FAIL,
                detail=f"jobs here ask for {wanted} cards, the machine has {cards}",
            )
        have = system.vram_mb / 1024
        if need and have < need:
            return Section(
                section="memory",
                verdict=Verdict.FAIL,
                detail=(
                    f"jobs here need {need} GB of card memory, the largest card holds {have:.0f}"
                ),
            )
        return Section(
            section="memory",
            verdict=Verdict.PASS,
            detail=f"{cards} cards, largest {have:.0f} GB"
            + (f", jobs need {need}" if need else ""),
        )

    def disk(self, system: System, role: Role) -> Section:
        """Whether the disk where the workspace lives has room for it."""
        need = _DISK[role]
        free = system.free_bytes / 1e9
        if system.free_bytes < need:
            return Section(
                section="disk",
                verdict=Verdict.WARN,
                detail=f"{free:.0f} GB free at {system.root}, a {role} wants {need / 1e9:.0f}",
            )
        return Section(
            section="disk", verdict=Verdict.PASS, detail=f"{free:.0f} GB free at {system.root}"
        )

    def tools(self, system: System) -> Section:
        """Whether every tool a center runs answers, and which versions it found."""
        missing = [tool for tool in _CENTER_TOOLS if tool not in system.tools]
        found = ", ".join(f"{name} {version}" for name, version in sorted(system.tools.items()))
        if missing:
            return Section(
                section="tools",
                verdict=Verdict.FAIL,
                detail=f"missing {', '.join(missing)}; found {found or 'nothing'}",
                fix="; ".join(install_command(system.system, tool) for tool in missing),
            )
        return Section(section="tools", verdict=Verdict.PASS, detail=found)

    def sync(self, system: System) -> Section:
        """Whether this machine can ship a mirror to a host: rsync, or the tar that replaces it."""
        carrier = next((tool for tool in ("rsync", "tar") if tool in system.tools), "")
        if not carrier:
            return Section(
                section="sync",
                verdict=Verdict.FAIL,
                detail="neither rsync nor tar answers, so no mirror can ship",
                fix=install_command(system.system, "tar"),
            )
        return Section(
            section="sync", verdict=Verdict.PASS, detail=f"mirrors ship through {carrier}"
        )

    def links(self, system: System) -> Section:
        """Whether the repository's symbolic links and deep paths survive a checkout here."""
        if not system.windows:
            return Section(
                section="links", verdict=Verdict.PASS, detail="symbolic links and long paths work"
            )
        notes = []
        fixes = []
        if system.symlinks:
            notes.append(
                "this account cannot create symbolic links, so linked directories become "
                "junctions and linked files hard links"
            )
            fixes.append(DEVELOPER_MODE)
        if not system.long_paths:
            notes.append("paths past 260 characters are refused")
            fixes.append(_LONG_PATHS)
        if notes:
            return Section(
                section="links",
                verdict=Verdict.WARN,
                detail="; ".join(notes),
                fix="; ".join(fixes),
            )
        return Section(
            section="links", verdict=Verdict.PASS, detail="symbolic links and long paths work"
        )

    def case(self, system: System) -> Section:
        """Whether the tracked tree checks out whole on this filesystem's idea of a name."""
        if system.case_sensitive:
            return Section(
                section="case", verdict=Verdict.PASS, detail="case-sensitive filesystem"
            )
        listed = Git(self.root).run("ls-files", "-z").stdout.split("\0")
        counted = Counter(path.casefold() for path in listed if path)
        clashes = sorted(path for path in listed if path and counted[path.casefold()] > 1)
        if not clashes:
            return Section(
                section="case",
                verdict=Verdict.PASS,
                detail="case-insensitive filesystem, and no tracked paths differ only in case",
            )
        unnamed = len(clashes) - _NAMED
        return Section(
            section="case",
            verdict=Verdict.FAIL,
            detail=(
                f"case-insensitive filesystem, and {len(clashes)} tracked paths differ only in "
                f"case: {', '.join(clashes[:_NAMED])}"
                + (f" and {unnamed} more" if unnamed > 0 else "")
            ),
            fix="rename one of each colliding pair in the repository",
        )

    def line_endings(self, system: System) -> Section:
        """Whether text checks out with the newline the repository stores."""
        automatic = system.git.get("core.autocrlf", "")
        attributes = self.root / ".gitattributes"
        pinned = attributes.is_file() and "eol=" in attributes.read_text(encoding="utf-8")
        if automatic == "true" and not pinned:
            return Section(
                section="line-endings",
                verdict=Verdict.WARN,
                detail="core.autocrlf=true rewrites every text file to CRLF on checkout",
                fix="git config --global core.autocrlf input",
            )
        how = (
            "the repository's .gitattributes"
            if pinned
            else f"core.autocrlf={automatic or 'unset'}"
        )
        return Section(
            section="line-endings", verdict=Verdict.PASS, detail=f"newlines follow {how}"
        )

    def shell(self, system: System) -> Section:
        """Whether the shells the agents run commands through are here."""
        found = ", ".join(sorted(system.shells))
        if system.windows and "bash" not in system.shells:
            return Section(
                section="shells",
                verdict=Verdict.WARN,
                detail=(
                    "no Git for Windows Bash, which Claude Code runs its commands in; "
                    f"found {found}"
                ),
                fix=install_command(system.system, "git"),
            )
        return Section(section="shells", verdict=Verdict.PASS, detail=found)

    def _floors(self, environment: str) -> dict[str, str]:
        """The floors `environment` solves against, its own over the workspace's."""
        try:
            own = self.manifest.environment(environment).system
        except MissionError:
            own = {}
        return own or self.manifest.system

    def _locked(self, environment: str) -> dict[str, list[str]] | None:
        """What `environment`'s lock installs per platform, read once; None when never solved."""
        if environment not in self.locks:
            path = Provisioner(self.root, self.manifest).pixi_for(environment).lock
            try:
                self.locks[environment] = packages(path.read_text(encoding="utf-8"))
            except FileNotFoundError:
                self.locks[environment] = None
        return self.locks[environment]

    def _built(self, platform: str, environment: str) -> Version | None:
        """The newest CUDA any build locked for `platform` was compiled against, None for none."""
        builds = (self._locked(environment) or {}).get(platform, [])
        found = [
            Version(f"{wheel[1]}.{wheel[2]}")
            for location in builds
            for wheel in _WHEEL_CUDA.finditer(location)
        ] + [Version(conda[1]) for location in builds for conda in _CONDA_CUDA.finditer(location)]
        return max(found, default=None)
