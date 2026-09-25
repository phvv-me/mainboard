"""Standard-library machine census, run here and sent to a remote Python over SSH stdin.

The one place a machine's operating system, shells, filesystem, git settings, tools and NVIDIA
driver are read. `facts` runs it in-process on whatever machine answers, and `center migrate`
sends this same file to a destination that has no Mainboard yet, so what a machine is found to
be never depends on which verb asked. It imports nothing outside the standard library for that
reason, and every answer is plain JSON data.
"""

import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

# How long one version or driver query may take. A tool that hangs on `--version` is a tool
# that does not answer, which is what an absent one says too.
_SECONDS = 20.0

# What a query that could not start or did not finish answers, the shape of a missing command.
_ABSENT = (127, "")

# The faults a query is allowed to end in: the program is not there, or it did not answer.
_FAULTS = (OSError, subprocess.SubprocessError)

# The first dotted version number in a tool's answer, `git version 2.51.0` naming `2.51.0`.
_VERSION = re.compile(r"\d+(?:\.\d+)+")

# The maximum CUDA version the driver supports, as the plain `nvidia-smi` banner prints it.
_DRIVER_CUDA = re.compile(r"CUDA Version:\s*([0-9.]+)")

# Every tool a report names, with the argv that makes it say its version. `git lfs` is asked
# through git, since its own binary is only ever reached as a git subcommand on Windows.
TOOLS: dict[str, tuple[str, ...]] = {
    "git": ("git", "--version"),
    "git-lfs": ("git", "lfs", "version"),
    "gh": ("gh", "--version"),
    "ssh": ("ssh", "-V"),
    "rsync": ("rsync", "--version"),
    "tar": ("tar", "--version"),
    "uv": ("uv", "--version"),
    "pixi": ("pixi", "--version"),
    "tectonic": ("tectonic", "--version"),
    "node": ("node", "--version"),
    "cargo": ("cargo", "--version"),
    "nvcc": ("nvcc", "--version"),
    "nvidia-smi": ("nvidia-smi", "--version"),
}

# The shells an agent or a person may run commands through, by the name each is known as.
SHELLS = ("bash", "zsh", "sh", "pwsh", "powershell", "cmd")

# The git settings a checkout depends on, read from the global scope a clone inherits.
GIT_SETTINGS = (
    "core.autocrlf",
    "core.eol",
    "core.symlinks",
    "core.longpaths",
    "credential.helper",
)

# Where Windows keeps its long-path switch and Developer Mode, the one that lets an ordinary
# account create a symbolic link.
_LONG_PATHS = ("HKLM\\SYSTEM\\CurrentControlSet\\Control\\FileSystem", "LongPathsEnabled")
_DEVELOPER_MODE = (
    "HKLM\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\AppModelUnlock",
    "AllowDevelopmentWithoutDevLicense",
)

# The Bash Git for Windows ships, the one a Claude Code session on Windows runs commands in.
_GIT_BASH = ("Git", "bin", "bash.exe")

type Runner = Callable[[Sequence[str]], tuple[int, str]]

# A plain JSON value, what every answer here is made of.
type Json = str | int | float | bool | None | Sequence[Json] | Mapping[str, Json]
type Finder = Callable[[str], str | None]


def run(command: Sequence[str]) -> tuple[int, str]:
    """Run `command` under a deadline, its stdout and stderr joined, answering rather than raising.

    command: the program and its arguments.
    """
    try:
        done = subprocess.run(
            list(command),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=_SECONDS,
            check=False,
        )
    except _FAULTS:
        return _ABSENT
    return done.returncode, done.stdout + done.stderr


class Census:
    """One machine's software as a plain JSON document, every question bounded and survivable.

    runner: runs one command and answers its status and output.
    finder: finds a program on PATH, None when it is not there.
    system: the platform as `platform.system()` spells it, this machine's when empty.
    """

    def __init__(
        self, runner: Runner = run, finder: Finder = shutil.which, system: str = ""
    ) -> None:
        self.runner = runner
        self.finder = finder
        self.system = system or platform.system()

    def survey(self, root: str) -> dict[str, Json]:
        """Everything this census reads, the filesystem measured where `root` lives.

        root: the workspace root, which need not exist yet; its nearest existing ancestor is
            what the filesystem questions are asked of.
        """
        anchor = self.anchor(root)
        cuda, gpus = self.nvidia()
        return {
            "system": self.system,
            "release": platform.release(),
            "version": self.version(),
            "arch": platform.machine(),
            "python": platform.python_version(),
            "shells": self.shells(),
            "root": str(anchor),
            "case_sensitive": self.case_sensitive(anchor),
            "symlinks": self.symlinks(anchor),
            "long_paths": self.long_paths(),
            "developer_mode": self.developer_mode(),
            "free_bytes": shutil.disk_usage(anchor).free,
            "git": self.git(),
            "tools": self.tools(),
            "cuda": cuda,
            "gpus": gpus,
        }

    @staticmethod
    def anchor(root: str) -> Path:
        """The nearest existing directory at or above `root`, where a clone would land."""
        path = Path(root).expanduser()
        while not path.is_dir() and path != path.parent:
            path = path.parent
        return path

    def version(self) -> str:
        """The operating system's own name for its version, a distribution's where it has one."""
        match self.system:
            case "Darwin":
                return f"macOS {platform.mac_ver()[0]}"
            case "Windows":
                release, build, _, _ = platform.win32_ver()
                return f"Windows {release} build {build}"
        try:
            text = Path("/etc/os-release").read_text(encoding="utf-8")
        except OSError:
            return platform.version()
        fields = dict(line.split("=", 1) for line in text.splitlines() if "=" in line)
        return fields.get("PRETTY_NAME", platform.version()).strip('"')

    def shells(self) -> dict[str, str]:
        """Every shell on PATH by name, with Git for Windows' Bash found where it installs."""
        found = {name: path for name in SHELLS if (path := self.finder(name))}
        if self.system == "Windows":
            found.pop("bash", None)
            if bash := self.git_bash():
                found["bash"] = bash
        return found

    def git_bash(self) -> str:
        """Git for Windows' Bash, never System32's WSL launcher that also answers to `bash`."""
        git = self.finder("git")
        roots = [Path(git).resolve().parent.parent] if git else []
        roots += [
            Path(value)
            for name in ("ProgramFiles", "ProgramW6432")
            if (value := os.environ.get(name))
        ]
        for root in roots:
            for candidate in (root.joinpath(*_GIT_BASH[1:]), root.joinpath(*_GIT_BASH)):
                if candidate.is_file():
                    return str(candidate)
        return ""

    @staticmethod
    def case_sensitive(anchor: Path) -> bool:
        """Whether two names differing only in case are two files where the workspace lives."""
        try:
            with tempfile.TemporaryDirectory(dir=anchor) as scratch:
                (Path(scratch) / "Probe").write_text("", encoding="utf-8")
                return not (Path(scratch) / "probe").exists()
        except OSError:
            return True

    @staticmethod
    def symlinks(anchor: Path) -> str:
        """Why a symbolic link cannot be made where the workspace lives, empty when it can."""
        try:
            with tempfile.TemporaryDirectory(dir=anchor) as scratch:
                Path(scratch, "link").symlink_to(Path(scratch, "target"))
        except OSError as refusal:
            return str(refusal) or type(refusal).__name__
        return ""

    def long_paths(self) -> bool:
        """Whether paths past 260 characters open here, a Windows switch and a given elsewhere."""
        return self.system != "Windows" or self.registry(*_LONG_PATHS) == 1

    def developer_mode(self) -> bool:
        """Whether Windows lets this account make symbolic links, False on every other system."""
        return self.system == "Windows" and self.registry(*_DEVELOPER_MODE) == 1

    def registry(self, key: str, value: str) -> int:
        """One DWORD out of the Windows registry, 0 when it is unset or unreadable."""
        status, said = self.runner(("reg", "query", key, "/v", value))
        found = re.search(r"REG_DWORD\s+0x([0-9a-fA-F]+)", said) if not status else None
        return int(found[1], 16) if found else 0

    def git(self) -> dict[str, str]:
        """The global git settings a clone inherits, each empty when unset."""
        return {
            name: said.strip() if not status else ""
            for name in GIT_SETTINGS
            for status, said in [self.runner(("git", "config", "--global", "--get", name))]
        }

    def tools(self) -> dict[str, str]:
        """Every tool that answers with a version, keyed by name; a silent one is left out."""
        found: dict[str, str] = {}
        for name, argv in TOOLS.items():
            if not self.finder(argv[0]):
                continue
            status, said = self.runner(argv)
            number = _VERSION.search(said)
            if not status and number:
                found[name] = number[0]
        return found

    def nvidia(self) -> tuple[str, list[Json]]:
        """The driver's maximum CUDA version and every NVIDIA card, both empty without one."""
        if not self.finder("nvidia-smi"):
            return "", []
        _, banner = self.runner(("nvidia-smi",))
        cuda = _DRIVER_CUDA.search(banner)
        status, listing = self.runner(
            (
                "nvidia-smi",
                "--query-gpu=name,driver_version,compute_cap,memory.total",
                "--format=csv,noheader,nounits",
            )
        )
        rows = [line.split(",") for line in listing.splitlines() if line.count(",") == 3]
        cards: list[Json] = [
            {
                "name": name.strip(),
                "driver": driver.strip(),
                "capability": capability.strip(),
                "vram_mb": int(memory.strip()) if memory.strip().isdigit() else 0,
            }
            for name, driver, capability, memory in rows
        ]
        return (cuda[1] if cuda else ""), (cards if not status else [])


def main(root: str) -> None:
    """Print this machine's census as one JSON line, what a remote caller reads back."""
    sys.stdout.write(json.dumps(Census().survey(root)) + "\n")
