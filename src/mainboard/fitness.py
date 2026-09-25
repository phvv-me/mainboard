# Whether one machine, as its census describes it, can serve this workspace: the census read
# against what the manifest and its lock ask for. It is the single judge behind the findings
# `facts`, `compute`, `setup`, `center verify` and `center migrate` show, so a driver too old for
# the lock is the same sentence wherever it surfaces.

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
from .workstation import DEVELOPER_MODE, abbreviated, install_command

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

# Free bytes needed where the workspace lives: a center holds the whole tree and every
# environment, a target one mirror.
_DISK = {"center": 60 * 10**9, "target": 20 * 10**9}

_CENTER_TOOLS = ("git", "git-lfs", "gh", "ssh")

_DRIVERS = "install a newer NVIDIA driver (https://www.nvidia.com/Download/index.aspx)"

# The Windows switch that lets paths past 260 characters open.
_LONG_PATHS = (
    "reg add HKLM\\SYSTEM\\CurrentControlSet\\Control\\FileSystem /v LongPathsEnabled "
    "/t REG_DWORD /d 1 /f (as administrator)"
)


class Role(StrEnum):
    """What a machine is for: the center that holds the monorepo, or a target that runs jobs."""

    CENTER = auto()
    TARGET = auto()


def _row(section: str, verdict: Verdict, detail: str, fix: str = "") -> Section:
    return Section(section=section, verdict=verdict, detail=detail, fix=fix)


class Fitness:
    """Judge one machine's census against this workspace's manifest and lock.

    root: the workspace root, where the lock and the tracked paths are read.
    """

    def __init__(self, root: Path, manifest: Manifest) -> None:
        self.root = root
        self.manifest = manifest
        self.locks: dict[str, dict[str, list[str]] | None] = {}

    def judge(
        self, system: System, *, host: str = "local", role: Role = Role.TARGET
    ) -> list[Section]:
        """Every finding about `system`, one row per question.

        A machine no census described is one row saying so, since every other question would
        only repeat that absence. A center is also asked about its tooling and its checkout.

        host: the alias whose profile says which environment and card the machine serves.
        """
        if not system.surveyed:
            fix = f"mainboard setup {host}" if host != "local" else ""
            return [
                _row("census", Verdict.WARN, "no software census recorded for this machine", fix)
            ]
        profile = self.manifest.profile(host)
        found = [
            self.platform(system),
            self.lock(system, profile.env),
            *self.cards(system, profile, profile.env),
            self.disk(system, role),
        ]
        if role is Role.CENTER:
            found += [
                self.tools(system),
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
            return _row(
                "platform",
                Verdict.PASS,
                f"{system.summary()} is {system.platform}, which the workspace declares",
            )
        return _row(
            "platform",
            Verdict.FAIL,
            f"{system.platform} is not among the declared platforms {declared}",
            f'add "{system.platform}" to [workspace] platforms, then mainboard install --resolve',
        )

    def lock(self, system: System, environment: str) -> Section:
        """Whether the lock this workspace solved holds builds for this machine's platform."""
        install = f"mainboard install {environment} --resolve"
        held = self._locked(environment)
        if held is None:
            return _row("lock", Verdict.WARN, f"{environment}: nothing solved yet", install)
        if not (builds := held.get(system.platform, [])):
            detail = f"{environment}: the lock holds no {system.platform} builds"
            return _row("lock", Verdict.FAIL, detail, install)
        detail = f"{environment}: {len(builds)} {system.platform} builds locked"
        return _row("lock", Verdict.PASS, detail)

    def cards(self, system: System, profile: HostProfile, environment: str) -> list[Section]:
        """The driver, the locked CUDA builds and the card memory, against what jobs need.

        A scheduler's ssh endpoint is its login node, whose cards (usually none) say nothing
        about the compute node a job lands on, so such a host answers once, with that caveat.
        """
        if profile.kind in _LOGIN_NODES:
            detail = f"{profile.kind} login node; cards are a compute node's, not read here"
            return [_row("cards", Verdict.PASS, detail)]
        floors = SystemFloors(declared=self._floors(environment)).on(system.platform)
        return [
            self.driver(system, floors.get("cuda", "")),
            self.builds(system, environment),
            self.memory(system, profile),
        ]

    def driver(self, system: System, floor: str) -> Section:
        """Whether the NVIDIA driver supports the CUDA floor the workspace solved against."""
        found = system.driver_cuda
        if not floor:
            return _row("driver", Verdict.PASS, f"no CUDA floor on {system.platform}")
        if found is None:
            detail = f"no NVIDIA driver answered, and the workspace solves CUDA {floor} builds"
            return _row("driver", Verdict.WARN, detail, _DRIVERS)
        if found < Version(floor):
            detail = f"the driver supports CUDA {found}, below the workspace floor {floor}"
            return _row("driver", Verdict.FAIL, detail, _DRIVERS)
        return _row("driver", Verdict.PASS, f"driver CUDA {found} meets the floor {floor}")

    def builds(self, system: System, environment: str) -> Section:
        """Whether the locked CUDA builds run on this driver and carry kernels for this card.

        CUDA's minor-version compatibility runs a build on any driver of its major version, the
        rule conda's `cuda-version X.Y` states as `__cuda >=X`, so only a newer major fails.
        """
        built = self._built(system.platform, environment)
        if built is None or not system.gpus:
            return _row("cuda-builds", Verdict.PASS, "no CUDA builds locked for this card to run")
        driver = system.driver_cuda
        if driver is not None and driver.major < built.major:
            detail = (
                f"builds locked for CUDA {built} need a CUDA {built.major} driver, not {driver}"
            )
            return _row("cuda-builds", Verdict.FAIL, detail, _DRIVERS)
        capability = system.capability
        needed = next(
            (cuda for generation, cuda in _KERNELS if capability and capability >= generation),
            None,
        )
        if needed is not None and built < needed:
            return _row(
                "cuda-builds",
                Verdict.FAIL,
                f"compute capability {capability} needs CUDA {needed} builds, "
                f"the lock holds CUDA {built}",
                f"raise the CUDA of the locked builds, then mainboard install {environment} "
                "--resolve",
            )
        detail = f"CUDA {built} builds run on this driver and card"
        if driver is not None and driver < built:
            detail = (
                f"CUDA {built} builds run on this CUDA {driver} driver by minor-version "
                "compatibility"
            )
        return _row("cuda-builds", Verdict.PASS, detail)

    def memory(self, system: System, profile: HostProfile) -> Section:
        """Whether the cards hold what the profile declares a job there needs."""
        need = profile.defaults.vram_gb
        cards = len(system.gpus)
        if (wanted := profile.defaults.gpus) > cards:
            detail = f"jobs here ask for {wanted} cards, the machine has {cards}"
            return _row("memory", Verdict.FAIL, detail)
        have = system.vram_mb / 1024
        if need and have < need:
            detail = f"jobs here need {need} GB of card memory, the largest card holds {have:.0f}"
            return _row("memory", Verdict.FAIL, detail)
        unified = " (unified with system memory)" if system.unified else ""
        detail = f"{cards} cards, largest {have:.0f} GB{unified}" + (
            f", jobs need {need}" if need else ""
        )
        return _row("memory", Verdict.PASS, detail)

    def disk(self, system: System, role: Role) -> Section:
        """Whether the disk where the workspace lives has room for it."""
        need = _DISK[role]
        free = f"{system.free_bytes / 1e9:.0f} GB free at {system.root}"
        if system.free_bytes < need:
            return _row("disk", Verdict.WARN, f"{free}, a {role} wants {need / 1e9:.0f}")
        return _row("disk", Verdict.PASS, free)

    def tools(self, system: System) -> Section:
        """Whether every tool a center runs answers, and which versions it found."""
        missing = [tool for tool in _CENTER_TOOLS if tool not in system.tools]
        found = ", ".join(f"{name} {version}" for name, version in sorted(system.tools.items()))
        if not missing:
            return _row("tools", Verdict.PASS, found)
        return _row(
            "tools",
            Verdict.FAIL,
            f"missing {', '.join(missing)}; found {found or 'nothing'}",
            "; ".join(install_command(system.system, tool) for tool in missing),
        )

    def links(self, system: System) -> Section:
        """Whether the repository's symbolic links and deep paths survive a checkout here."""
        notes = []
        fixes = []
        if system.windows and system.symlinks:
            notes.append(
                "this account cannot create symbolic links, so linked directories become "
                "junctions and linked files hard links"
            )
            fixes.append(DEVELOPER_MODE)
        if system.windows and not system.long_paths:
            notes.append("paths past 260 characters are refused")
            fixes.append(_LONG_PATHS)
        if not notes:
            return _row("links", Verdict.PASS, "symbolic links and long paths work")
        return _row("links", Verdict.WARN, "; ".join(notes), "; ".join(fixes))

    def case(self, system: System) -> Section:
        """Whether the tracked tree checks out whole on this filesystem's idea of a name."""
        if system.case_sensitive:
            return _row("case", Verdict.PASS, "case-sensitive filesystem")
        listed = [path for path in Git(self.root).run("ls-files", "-z").stdout.split("\0") if path]
        counted = Counter(path.casefold() for path in listed)
        clashes = sorted(path for path in listed if counted[path.casefold()] > 1)
        if not clashes:
            detail = "case-insensitive filesystem, and no tracked paths differ only in case"
            return _row("case", Verdict.PASS, detail)
        return _row(
            "case",
            Verdict.FAIL,
            f"case-insensitive filesystem, and {len(clashes)} tracked paths differ only in "
            f"case: {abbreviated(clashes)}",
            "rename one of each colliding pair in the repository",
        )

    def line_endings(self, system: System) -> Section:
        """Whether text checks out with the newline the repository stores."""
        automatic = system.git.get("core.autocrlf", "")
        attributes = self.root / ".gitattributes"
        pinned = attributes.is_file() and "eol=" in attributes.read_text(encoding="utf-8")
        if automatic == "true" and not pinned:
            return _row(
                "line-endings",
                Verdict.WARN,
                "core.autocrlf=true rewrites every text file to CRLF on checkout",
                "git config --global core.autocrlf input",
            )
        how = (
            "the repository's .gitattributes"
            if pinned
            else f"core.autocrlf={automatic or 'unset'}"
        )
        return _row("line-endings", Verdict.PASS, f"newlines follow {how}")

    def shell(self, system: System) -> Section:
        """Whether the shells the agents run commands through are here."""
        found = ", ".join(sorted(system.shells))
        if system.windows and "bash" not in system.shells:
            detail = (
                f"no Git for Windows Bash, which Claude Code runs its commands in; found {found}"
            )
            return _row("shells", Verdict.WARN, detail, install_command(system.system, "git"))
        return _row("shells", Verdict.PASS, found)

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
