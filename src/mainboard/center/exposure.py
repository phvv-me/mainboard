# The default environment's tools, reachable from every shell an agent opens on the center.
#
# Agents rarely run inside an activated environment: Claude Code runs its commands through the
# user's zsh or, on Windows, Git Bash; Codex through PowerShell or a login shell; opencode through
# whatever shell it spawns. The portable toolset the manifest puts in the default environment
# (the uutils coreutils, ripgrep, fd, sd, yq and the rest) only standardizes their commands if each
# of those shells finds it first, so the environment's executable directories go on the one PATH
# each platform gives every new process:
#
# - Windows keeps a user PATH in the registry, which cmd, PowerShell and Git Bash all start from,
#   so the directories are prepended there, and every entry an older prefix left is replaced.
# - macOS and Linux have no such store, only each shell's startup files, so one generated file,
#   `~/.config/mainboard/path.sh`, puts the directories on PATH, and a marked line sources it from
#   the startup file every mode of each shell reads: `~/.zshenv` for zsh (login or not,
#   interactive or not), and `~/.profile` and `~/.bashrc` for bash and sh.
#
# Each shell kind is then started the way an agent starts it, from a fresh environment, and asked
# where each tool resolves, which is the only proof that counts.

import json
import os
from pathlib import Path
from shlex import quote as posix_quote
from typing import TYPE_CHECKING

from ..core.project import Project
from ..core.section import Section, Verdict
from ..dispatch.shells import quoted

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

# Runs one command under an environment, answering its status and joined output.
type Spawn = Callable[[Sequence[str], Mapping[str, str]], tuple[int, str]]

# Where conda puts executables in a Windows prefix, in the order its own activation searches.
_WINDOWS_DIRS = ("", "Library/mingw-w64/bin", "Library/usr/bin", "Library/bin", "Scripts", "bin")

# The file extensions Windows runs by name.
_RUNNABLE = (".exe", ".bat", ".cmd")

# The generated PATH file on macOS and Linux, and the marked line that sources it.
PATH_FILE = ".config/mainboard/path.sh"
_BEGIN = "# >>> mainboard >>>"
_END = "# <<< mainboard <<<"
_SOURCE = f'[ -f "$HOME/{PATH_FILE}" ] && . "$HOME/{PATH_FILE}"'

# Which startup file each POSIX shell reads in every mode an agent starts it in.
_STARTUP = {"zsh": (".zshenv",), "bash": (".profile", ".bashrc"), "sh": (".profile",)}

# The PATH a fresh POSIX login starts from, before any startup file adds to it.
_BARE_PATH = "/usr/bin:/bin:/usr/sbin:/sbin"

# A path segment only a Mainboard environment prefix carries, which marks a stale user PATH entry.
_PREFIX_MARK = os.sep.join(("", ".pixi", "envs", ""))

# This tool, which every agent shell must reach wherever uv put it.
_TOOL = Project().name

# How many names a detail line spells out before it only counts the rest.
_NAMED = 6


def directories(prefix: Path, system: str) -> list[Path]:
    """Every directory of `prefix` that holds executables, in the order PATH should carry them.

    prefix: the installed environment prefix.
    system: the platform as `platform.system()` spells it.
    """
    if system == "Windows":
        return [prefix / relative if relative else prefix for relative in _WINDOWS_DIRS]
    return [prefix / "bin"]


def executables(prefix: Path, packages: Sequence[str], system: str) -> list[str]:
    """The command names the declared `packages` put in `prefix`, from conda's own records.

    prefix: the installed environment prefix.
    packages: the conda package names the manifest declares.
    system: the platform as `platform.system()` spells it.
    """
    wanted = set(packages)
    names: set[str] = set()
    for record in sorted((prefix / "conda-meta").glob("*.json")):
        document = json.loads(record.read_text(encoding="utf-8"))
        if document.get("name") not in wanted:
            continue
        for file in document.get("files", []):
            path = Path(file)
            if system == "Windows" and path.suffix.lower() in _RUNNABLE:
                names.add(path.stem)
            elif (
                system != "Windows"
                and path.parent.as_posix() == "bin"
                and _runnable(prefix / path)
            ):
                names.add(path.name)
    return sorted(names)


class Exposure:
    """Put an environment's executable directories on every shell's PATH, then prove it.

    folders: the environment's executable directories, first searched first.
    system: the platform as `platform.system()` spells it.
    home: the user's home directory, where startup files and the PATH file live.
    shells: every shell on this machine, by name, with its path.
    spawn: runs one command under an environment.
    """

    def __init__(
        self,
        folders: Sequence[Path],
        *,
        system: str,
        home: Path,
        shells: Mapping[str, str],
        spawn: Spawn,
    ) -> None:
        self.folders = [str(folder) for folder in folders]
        self.system = system
        self.home = home
        self.shells = dict(shells)
        self.spawn = spawn

    def apply(self) -> Section:
        """Make the directories reachable, changing only what is not already so."""
        return self._registry() if self.system == "Windows" else self._startup()

    def verify(self, names: Sequence[str]) -> list[Section]:
        """Where each of `names`, and this tool, resolves from every shell here, one row each.

        A name that resolves outside the environment is shadowed, a PowerShell alias of the same
        name included, and one that resolves nowhere is missing; both are named with the fix.
        This tool lives in its own uv environment, so it only has to resolve at all.
        """
        asked = [_TOOL, *names]
        return [
            self._judged(kind, names, self._resolved(kind, path, asked))
            for kind, path in sorted(self.shells.items())
        ]

    def _registry(self) -> Section:
        """Prepend the directories to the Windows user PATH, dropping older prefixes' entries."""
        _, held = self.spawn(
            (
                "powershell",
                "-NoProfile",
                "-Command",
                "[Environment]::GetEnvironmentVariable('Path','User')",
            ),
            os.environ,
        )
        entries = [entry for entry in held.strip().split(";") if entry]
        kept = [
            entry
            for entry in entries
            if entry not in self.folders and _PREFIX_MARK not in entry.replace("/", os.sep)
        ]
        wanted = [*self.folders, *kept]
        if wanted == entries:
            return Section(
                section="path", verdict=Verdict.PASS, detail="the environment is on the user PATH"
            )
        value = ";".join(wanted)
        status, said = self.spawn(
            (
                "powershell",
                "-NoProfile",
                "-Command",
                f"[Environment]::SetEnvironmentVariable('Path', {quoted(value)}, 'User')",
            ),
            os.environ,
        )
        if status:
            return Section(
                section="path",
                verdict=Verdict.FAIL,
                detail=f"the user PATH could not be written: {said.strip()[-160:]}",
                fix=f"[Environment]::SetEnvironmentVariable('Path', {quoted(value)}, 'User')",
            )
        return Section(
            section="path",
            verdict=Verdict.PASS,
            detail=(
                f"put {len(self.folders)} environment directories on the user PATH; "
                "new shells see them"
            ),
        )

    def _startup(self) -> Section:
        """Write the PATH file and source it from each present shell's startup files."""
        path_file = self.home / PATH_FILE
        lines = [
            "# Written by `mainboard center verify`: the default environment and the tool on PATH",
            *(
                _prepended(folder)
                for folder in reversed([*self.folders, str(self.home / ".local" / "bin")])
            ),
            "export PATH",
        ]
        changed = _written(path_file, "\n".join(lines) + "\n")
        files = sorted({file for kind in self.shells for file in _STARTUP.get(kind, ())})
        wired = [file for file in files if _sourced(self.home / file)]
        touched = [path_file.name] if changed else []
        touched += wired
        return Section(
            section="path",
            verdict=Verdict.PASS,
            detail=(
                f"wrote {', '.join(touched)}"
                if touched
                else "the environment is on every shell's PATH"
            )
            + (f"; startup files carry the {_BEGIN} line" if files else ""),
        )

    def _resolved(self, kind: str, path: str, names: Sequence[str]) -> dict[str, tuple[str, str]]:
        """Each name's resolution in one shell kind: its kind of command and where it lives.

        A PowerShell alias has no source, so its definition, the command it stands for, is
        where it lives; the row then reads as the shadow it is rather than as missing.
        """
        environment = self._fresh(kind)
        match kind:
            case "powershell" | "pwsh":
                listed = ",".join(quoted(name) for name in names)
                script = (
                    f"foreach ($t in @({listed})) {{ "
                    "$c = Get-Command $t -ErrorAction SilentlyContinue | Select-Object -First 1; "
                    "$w = if ($c.Source) { $c.Source } else { $c.Definition }; "
                    '"$t`t$($c.CommandType)`t$w" }'
                )
                _, said = self.spawn((path, "-NoProfile", "-Command", script), environment)
            case "cmd":
                _, said = self.spawn((path, "/d", "/c", "where", *names), environment)
                found: dict[str, tuple[str, str]] = {}
                for line in said.splitlines():
                    stem = Path(line.strip()).stem
                    if stem in names and stem not in found:
                        found[stem] = ("Application", line.strip())
                return found
            case _:
                loop = " ".join(posix_quote(name) for name in names)
                script = (
                    f'for t in {loop}; do p=$(command -v "$t"); '
                    'case "$p" in /*) k=Application ;; *) k=Builtin ;; esac; '
                    'printf "%s\\t%s\\t%s\\n" "$t" "$k" "$p"; done'
                )
                flag = "-c" if kind == "zsh" else "-lc"
                _, said = self.spawn((path, flag, script), environment)
        rows = [line.split("\t") for line in said.splitlines() if line.count("\t") == 2]
        return {name: (sort, where) for name, sort, where in rows if where}

    def _judged(
        self, kind: str, names: Sequence[str], found: Mapping[str, tuple[str, str]]
    ) -> Section:
        """One shell's row: how many names resolve into the environment, and which do not."""
        inside = [name for name in names if name in found and self._ours(found[name][1])]
        missing = [name for name in (_TOOL, *names) if name not in found]
        builtins = [name for name in names if found.get(name, ("", ""))[0] == "Builtin"]
        shadowed = [
            f"{name} ({found[name][0].lower()})"
            for name in names
            if name in found and name not in builtins and not self._ours(found[name][1])
        ]
        detail = f"{len(inside)} of {len(names)} resolve into the environment"
        if builtins:
            detail += f", {len(builtins)} are the shell's own builtins"
        if missing:
            detail += f"; missing {_listed(missing)}"
        if shadowed:
            detail += f"; shadowed by {_listed(shadowed)}"
        if not missing and not shadowed:
            return Section(section=f"path {kind}", verdict=Verdict.PASS, detail=detail)
        aliases = [name for name in names if name in found and found[name][0] == "Alias"]
        fix = (
            f"add `Remove-Item Alias:{',Alias:'.join(aliases)} -Force "
            "-ErrorAction SilentlyContinue` to $PROFILE"
            if aliases
            else "mainboard install, then open a new shell"
        )
        return Section(section=f"path {kind}", verdict=Verdict.WARN, detail=detail, fix=fix)

    def _ours(self, where: str) -> bool:
        """Whether a resolved path lies in one of the environment's directories."""
        folded = where.replace("\\", "/").casefold()
        return any(
            folded.startswith(folder.replace("\\", "/").casefold().rstrip("/") + "/")
            for folder in self.folders
        )

    def _fresh(self, kind: str) -> dict[str, str]:
        """The environment a new shell of `kind` starts from, before its startup files run.

        On Windows that is the machine PATH and the user PATH as the registry holds them now,
        which this process, started before any change, does not carry itself.
        """
        if self.system == "Windows":
            _, joined = self.spawn(
                (
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    "[Environment]::GetEnvironmentVariable('Path','Machine') + ';' + "
                    "[Environment]::GetEnvironmentVariable('Path','User')",
                ),
                os.environ,
            )
            return {**os.environ, "PATH": joined.strip()}
        return {
            "HOME": str(self.home),
            "USER": os.environ.get("USER", ""),
            "PATH": _BARE_PATH,
            "SHELL": self.shells.get(kind, ""),
            "TERM": "dumb",
        }


def _runnable(path: Path) -> bool:
    """Whether a file a package declares can be run by name, or is gone and so worth naming."""
    return not path.exists() or os.access(path, os.X_OK)


def _written(path: Path, text: str) -> bool:
    """Write `text` to `path` unless it already holds it, answering whether it changed."""
    try:
        if path.read_text(encoding="utf-8") == text:
            return False
    except FileNotFoundError:
        path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return True


def _sourced(startup: Path) -> bool:
    """Append the marked line sourcing the PATH file to `startup`, answering whether it did."""
    try:
        text = startup.read_text(encoding="utf-8")
    except FileNotFoundError:
        text = ""
    if _BEGIN in text:
        return False
    block = f"{_BEGIN}\n{_SOURCE}\n{_END}\n"
    startup.write_text(
        text + ("\n" if text and not text.endswith("\n") else "") + block, encoding="utf-8"
    )
    return True


def _prepended(folder: str) -> str:
    """The POSIX line putting `folder` at the front of PATH once, however often it is sourced."""
    held = posix_quote(folder)
    return f'case ":$PATH:" in *:{held}:*) ;; *) PATH={held}:"$PATH" ;; esac'


def _listed(names: Sequence[str]) -> str:
    """The first few names, and how many more there are."""
    rest = len(names) - _NAMED
    return ", ".join(names[:_NAMED]) + (f" and {rest} more" if rest > 0 else "")
