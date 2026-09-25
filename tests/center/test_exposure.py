import json
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from enum import StrEnum, auto
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from mainboard.center.exposure import PATH_FILE, Exposure, directories, executables
from mainboard.core.section import Verdict

from ..strategies import WORDS

# The name this tool answers to, which every shell must resolve on top of the declared tools.
_TOOL = "mainboard"

# The PATH value a PowerShell `SetEnvironmentVariable` call writes, as its single-quoted literal.
_WRITTEN = re.compile(r"'Path', '((?:[^']|'')*)', 'User'")


class Resolution(StrEnum):
    """Where one name resolves in a shell, as the shell's own report would say it."""

    INSIDE = auto()
    OUTSIDE = auto()
    BUILTIN = auto()
    ALIAS = auto()
    MISSING = auto()


# What each shell kind can report: POSIX shells know builtins, PowerShell knows aliases, and
# cmd's `where` lists only files on disk.
_REPORTABLE = {
    "zsh": (Resolution.INSIDE, Resolution.OUTSIDE, Resolution.BUILTIN, Resolution.MISSING),
    "bash": (Resolution.INSIDE, Resolution.OUTSIDE, Resolution.BUILTIN, Resolution.MISSING),
    "sh": (Resolution.INSIDE, Resolution.OUTSIDE, Resolution.BUILTIN, Resolution.MISSING),
    "powershell": (Resolution.INSIDE, Resolution.OUTSIDE, Resolution.ALIAS, Resolution.MISSING),
    "pwsh": (Resolution.INSIDE, Resolution.OUTSIDE, Resolution.ALIAS, Resolution.MISSING),
    "cmd": (Resolution.INSIDE, Resolution.OUTSIDE, Resolution.MISSING),
}


class Machine:
    """A scripted process seam: a registry-backed user PATH and shells that report resolutions.

    user: the user PATH the registry holds, rewritten by a successful set.
    machine: the machine PATH the registry holds.
    refusal: what a PATH write says when it fails, empty for a write that succeeds.
    shells: each shell path's report, the text its resolution script prints.
    """

    def __init__(
        self,
        *,
        user: str = "",
        machine: str = "C:\\Windows\\system32",
        refusal: str = "",
        shells: Mapping[str, str] | None = None,
    ) -> None:
        self.user = user
        self.machine = machine
        self.refusal = refusal
        self.shells = dict(shells or {})
        self.ran: list[tuple[tuple[str, ...], dict[str, str]]] = []

    def __call__(self, command: Sequence[str], environment: Mapping[str, str]) -> tuple[int, str]:
        self.ran.append((tuple(command), dict(environment)))
        script = command[-1]
        if "[Environment]::SetEnvironmentVariable" in script:
            if self.refusal:
                return 1, f"{self.refusal}\r\n"
            found = _WRITTEN.search(script)
            assert found is not None
            self.user = found.group(1).replace("''", "'")
            return 0, ""
        if "'Machine'" in script:
            return 0, f"{self.machine};{self.user}\r\n"
        if "[Environment]::GetEnvironmentVariable" in script:
            return 0, f"{self.user}\r\n"
        return 0, self.shells[command[0]]

    @property
    def writes(self) -> list[tuple[str, ...]]:
        """Every command that changed the registry."""
        return [argv for argv, _ in self.ran if "SetEnvironmentVariable" in argv[-1]]


def report(kind: str, folder: Path, table: Mapping[str, Resolution]) -> str:
    """What one shell kind prints for `table`, in the exact shape its resolution script uses."""
    lines: list[str] = []
    for name, where in table.items():
        match kind, where:
            case "cmd", Resolution.INSIDE:
                lines += [str(folder / f"{name}.exe"), str(Path("elsewhere") / f"{name}.exe")]
            case "cmd", Resolution.OUTSIDE:
                lines.append(str(Path("elsewhere") / f"{name}.exe"))
            case "cmd", _:
                lines.append("INFO: Could not find files for the given pattern(s).")
            case "powershell" | "pwsh", Resolution.INSIDE:
                lines.append(f"{name}\tApplication\t{str(folder).upper()}\\{name}.EXE")
            case "powershell" | "pwsh", Resolution.OUTSIDE:
                lines.append(f"{name}\tApplication\tC:\\Windows\\{name}.exe")
            case "powershell" | "pwsh", Resolution.ALIAS:
                lines.append(f"{name}\tAlias\tGet-ChildItem")
            case "powershell" | "pwsh", _:
                lines.append(f"{name}\t\t")
            case _, Resolution.INSIDE:
                lines.append(f"{name}\tApplication\t{folder}/{name}")
            case _, Resolution.OUTSIDE:
                lines.append(f"{name}\tApplication\t/usr/bin/{name}")
            case _, Resolution.BUILTIN:
                lines.append(f"{name}\tBuiltin\t{name}")
            case _:
                lines.append(f"{name}\tBuiltin\t")
    return "\n".join(lines) + "\n"


def exposed(
    folders: Sequence[Path], system: str, home: Path, spawn: Machine, **shells: str
) -> Exposure:
    """An exposure of `folders` on a `system` machine whose shells are `shells`."""
    return Exposure(folders, system=system, home=home, shells=shells, spawn=spawn)


@given(
    kind=st.sampled_from(sorted(_REPORTABLE)),
    system=st.sampled_from(["Windows", "Linux"]),
    names=st.lists(WORDS, unique=True, max_size=8),
    data=st.data(),
)
def test_every_shell_row_counts_the_tools_inside_and_names_each_that_is_not(
    tmp_path: Path, kind: str, system: str, names: list[str], data: st.DataObject
) -> None:
    """One property over every shell kind and every way a name can resolve there.

    A row passes only when the tool resolves somewhere and every declared name resolves into
    the environment or is the shell's own builtin; anything else is a warning naming the name,
    with the alias removal as the fix whenever PowerShell shadows a name with an alias. Each
    shell is started from a fresh environment: the registry's PATH on Windows, a bare login
    elsewhere, never this process's own.
    """
    folder = tmp_path / "env" / "bin"
    allowed = _REPORTABLE[kind]
    table = {name: data.draw(st.sampled_from(allowed)) for name in names}
    tool = data.draw(st.sampled_from([r for r in allowed if r is not Resolution.BUILTIN]))
    if tool is Resolution.ALIAS:
        tool = Resolution.OUTSIDE
    said = report(kind, folder, {_TOOL: tool, **table})
    spawn = Machine(user="C:\\user", shells={f"/shells/{kind}": said})

    (row,) = exposed([folder], system, tmp_path, spawn, **{kind: f"/shells/{kind}"}).verify(names)

    missing = [_TOOL] * (tool is Resolution.MISSING) + [
        name for name, where in table.items() if where is Resolution.MISSING
    ]
    shadowed = [n for n, w in table.items() if w in (Resolution.OUTSIDE, Resolution.ALIAS)]
    inside = sum(where is Resolution.INSIDE for where in table.values())
    assert row.section == f"path {kind}"
    assert row.detail.startswith(f"{inside} of {len(names)} resolve into the environment")
    assert ("; missing " in row.detail) == bool(missing)
    assert ("; shadowed by " in row.detail) == bool(shadowed)
    assert ("builtins" in row.detail) == (Resolution.BUILTIN in table.values())
    assert all(name in row.detail for name in [*missing, *shadowed][:6])
    assert row.verdict == (Verdict.WARN if missing or shadowed else Verdict.PASS)
    aliases = [name for name, where in table.items() if where is Resolution.ALIAS]
    if row.verdict == Verdict.WARN:
        assert ("Remove-Item Alias:" in row.fix) == bool(aliases)
    (argv, environment) = spawn.ran[-1]
    assert argv[0] == f"/shells/{kind}"
    if system == "Windows":
        assert environment["PATH"] == "C:\\Windows\\system32;C:\\user"
    else:
        assert environment == {
            "HOME": str(tmp_path),
            "USER": os.environ.get("USER", ""),
            "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
            "SHELL": f"/shells/{kind}",
            "TERM": "dumb",
        }


def test_each_shell_kind_is_asked_the_way_an_agent_starts_it(tmp_path: Path) -> None:
    """zsh runs `-c` since `.zshenv` covers every mode; bash and sh need `-lc` for `.profile`.

    PowerShell asks `Get-Command`, falling back to an alias's definition since a built-in alias
    has no source, and cmd asks `where`, each for the tool and every name. The
    rows come back sorted by shell, and a long list is cut to six names and a count.
    """
    folder = tmp_path / "env" / "bin"
    names = [f"tool{index}" for index in range(8)]
    table = dict.fromkeys(names, Resolution.MISSING)
    shells = {
        "bash": "/bin/bash",
        "zsh": "/bin/zsh",
        "pwsh": "/bin/pwsh",
        "cmd": "/bin/cmd",
    }
    spawn = Machine(shells={path: report(kind, folder, table) for kind, path in shells.items()})

    rows = exposed([folder], "Linux", tmp_path, spawn, **shells).verify(names)

    assert [row.section for row in rows] == ["path bash", "path cmd", "path pwsh", "path zsh"]
    assert rows[0].detail == (
        "0 of 8 resolve into the environment; "
        "missing mainboard, tool0, tool1, tool2, tool3, tool4 and 3 more"
    )
    assert rows[0].fix == "mainboard install, then open a new shell"
    asked = {argv[0]: argv for argv, _ in spawn.ran}
    assert asked["/bin/bash"][1] == "-lc"
    assert asked["/bin/zsh"][1] == "-c"
    assert "for t in mainboard tool0" in asked["/bin/zsh"][2]
    assert asked["/bin/cmd"] == ("/bin/cmd", "/d", "/c", "where", _TOOL, *names)
    assert asked["/bin/pwsh"][1:3] == ("-NoProfile", "-Command")
    assert "@('mainboard','tool0'" in asked["/bin/pwsh"][3]
    assert "if ($c.Source) { $c.Source } else { $c.Definition }" in asked["/bin/pwsh"][3]


def test_a_powershell_alias_is_named_with_the_profile_line_that_removes_it(
    tmp_path: Path,
) -> None:
    """PowerShell's `curl` and `ls` aliases beat any file on PATH, so the fix is the profile.

    A built-in alias has no source, so the script reports its definition in that column, and
    the row reads as a shadow rather than as a missing tool.
    """
    folder = tmp_path / "env"
    table = {_TOOL: Resolution.INSIDE, "curl": Resolution.ALIAS, "ls": Resolution.ALIAS}
    spawn = Machine(shells={"pwsh": report("pwsh", folder, table)})

    (row,) = exposed([folder], "Windows", tmp_path, spawn, pwsh="pwsh").verify(["curl", "ls"])

    assert row.verdict == Verdict.WARN
    assert row.detail == (
        "0 of 2 resolve into the environment; shadowed by curl (alias), ls (alias)"
    )
    assert row.fix == (
        "add `Remove-Item Alias:curl,Alias:ls -Force -ErrorAction SilentlyContinue` to $PROFILE"
    )


def entries(data: st.DataObject, folders: list[str], root: Path) -> tuple[list[str], list[str]]:
    """A user PATH mixing the environment's folders, stale prefixes and unrelated entries."""
    others = [str(root / "other" / word) for word in data.draw(st.lists(WORDS, unique=True))]
    stale = [
        str(root / "old" / ".pixi" / "envs" / word / "bin")
        for word in data.draw(st.lists(WORDS, unique=True, max_size=3))
    ]
    present = data.draw(st.sets(st.sampled_from(folders)))
    held = data.draw(st.permutations([*present, *stale, *others]))
    return held, [entry for entry in held if entry in others]


@given(data=st.data())
def test_the_user_path_puts_the_environment_first_and_keeps_every_other_entry(
    tmp_path: Path, data: st.DataObject
) -> None:
    """Whatever the registry held, the environment's folders end up first, in their own order.

    Every entry an older prefix left is dropped, every unrelated entry is kept in its order, and
    a second apply against what the first wrote changes nothing, so running verify again is free.
    """
    folders = directories(tmp_path / "prefix", "Windows")
    spelled = [str(folder) for folder in folders]
    held, others = entries(data, spelled, tmp_path)
    spawn = Machine(user=";".join(["", *held, ""]))
    exposure = exposed(folders, "Windows", tmp_path, spawn)

    first = exposure.apply()
    second = exposure.apply()

    assert [entry for entry in spawn.user.split(";") if entry] == [*spelled, *others]
    assert first.verdict == second.verdict == Verdict.PASS
    assert len(spawn.writes) == (0 if held == [*spelled, *others] else 1)
    assert second.detail == "the environment is on the user PATH"


def test_a_user_path_that_cannot_be_written_is_a_failure_with_the_command_to_run(
    tmp_path: Path,
) -> None:
    """A refused registry write says why, and hands over the exact PowerShell line to retry."""
    folders = [tmp_path / "prefix"]
    spawn = Machine(user="C:\\keep;it's", refusal="Access is denied.")

    row = exposed(folders, "Windows", tmp_path, spawn).apply()

    assert row.verdict == Verdict.FAIL
    assert row.detail == "the user PATH could not be written: Access is denied."
    assert row.fix == (
        f"[Environment]::SetEnvironmentVariable('Path', '{folders[0]};C:\\keep;it''s', 'User')"
    )


def test_posix_startup_files_are_wired_once_for_the_shells_present(home: Path) -> None:
    """The PATH file and one marked line per startup file are written once and never again.

    Only the files the present shells read are touched, text already there is kept, and a file
    that ends without a newline gets one before the marked block, so nothing is glued together.
    """
    (home / ".profile").write_text("export EDITOR=vi", encoding="utf-8")
    (home / ".zshenv").write_text("setopt nobeep\n", encoding="utf-8")
    folders = [home / "env" / "bin"]
    exposure = exposed(folders, "Darwin", home, Machine(), bash="/bin/bash", fish="/bin/fish")

    first = exposure.apply()
    second = exposure.apply()

    marked = '# >>> mainboard >>>\n[ -f "$HOME/.config/mainboard/path.sh" ]'
    carried = "; startup files carry the # >>> mainboard >>> line"
    assert first.detail == f"wrote path.sh, .bashrc, .profile{carried}"
    assert second.detail == f"the environment is on every shell's PATH{carried}"
    assert (
        (home / ".profile").read_text(encoding="utf-8").startswith(f"export EDITOR=vi\n{marked}")
    )
    assert (home / ".bashrc").read_text(encoding="utf-8").startswith(marked)
    assert (home / ".zshenv").read_text(encoding="utf-8") == "setopt nobeep\n"
    assert first.verdict == second.verdict == Verdict.PASS


def test_a_machine_with_no_known_shell_still_gets_the_path_file(home: Path) -> None:
    """Without a shell whose startup files are known, only the PATH file is written.

    It is rewritten only when the environment moved, so an unchanged one is left alone.
    """
    exposure = exposed([home / "env" / "bin"], "Linux", home, Machine(), fish="/bin/fish")
    moved = exposed([home / "new" / "bin"], "Linux", home, Machine(), fish="/bin/fish")

    assert exposure.apply().detail == "wrote path.sh"
    assert exposure.apply().detail == "the environment is on every shell's PATH"
    assert moved.apply().detail == "wrote path.sh"
    assert str(home / "new" / "bin") in (home / PATH_FILE).read_text(encoding="utf-8")
    assert sorted(path.name for path in home.iterdir()) == [".config"]


@pytest.mark.skipif(
    sys.platform == "win32" or shutil.which("sh") is None,
    reason="the PATH file is POSIX shell, sourced only on macOS and Linux",
)
def test_the_path_file_puts_the_environment_first_once_however_often_it_is_sourced(
    home: Path,
) -> None:
    """The generated file is real shell: sourced twice, each folder is on PATH exactly once.

    The environment's folders come first in their own order, then the tool's `~/.local/bin`,
    then whatever PATH the shell started with, and a folder with a space survives quoting.
    """
    folders = [home / "my env" / "bin", home / "env" / "Library" / "bin"]
    exposed(folders, "Linux", home, Machine(), zsh="/bin/zsh").apply()
    path_file = home / PATH_FILE

    done = subprocess.run(
        ["sh", "-c", f'. "{path_file}"; . "{path_file}"; printf %s "$PATH"'],
        env={"PATH": "/usr/bin:/bin", "HOME": str(home)},
        capture_output=True,
        text=True,
        check=True,
    )

    expected = [*map(str, folders), str(home / ".local" / "bin"), "/usr/bin", "/bin"]
    assert done.stdout.split(":") == expected


def test_the_executable_folders_follow_conda_s_own_activation_order(tmp_path: Path) -> None:
    """Windows prefixes spread executables over six folders, the prefix itself first."""
    windows = ["Library/mingw-w64/bin", "Library/usr/bin", "Library/bin", "Scripts", "bin"]
    assert directories(tmp_path, "Windows") == [tmp_path, *(tmp_path / sub for sub in windows)]
    assert directories(tmp_path, "Linux") == directories(tmp_path, "Darwin") == [tmp_path / "bin"]


def test_the_commands_a_package_installs_are_read_from_conda_s_records(tmp_path: Path) -> None:
    """Only declared packages count, and each platform reads its own runnable files.

    Windows runs `.exe`, `.bat` and `.cmd` from anywhere in the prefix, by stem and whatever the
    case. POSIX runs the files directly in `bin/` that are executable, and a file the record
    names but the disk lacks is kept, since its absence is exactly what a row should name.
    """
    meta = tmp_path / "conda-meta"
    meta.mkdir()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "rg").touch(mode=0o755)
    plain = ["bin/rg", "bin/gone", "bin/sub/nested", "share/man/rg.1"]
    if sys.platform != "win32":
        (bin_dir / "readme").touch(mode=0o644)
        plain.append("bin/readme")
    windows = ["Library/bin/fd.exe", "Scripts/act.BAT", "Library/bin/yq.cmd", "Library/lib.dll"]
    records = {
        "ripgrep-14.json": {"name": "ripgrep", "files": plain},
        "fd-10.json": {"name": "fd", "files": windows},
        "empty-1.json": {"name": "empty"},
        "other-1.json": {"name": "other", "files": ["bin/other", "other.exe"]},
    }
    for file, record in records.items():
        (meta / file).write_text(json.dumps(record), encoding="utf-8")
    declared = ["ripgrep", "fd", "empty", "absent"]

    assert executables(tmp_path, declared, "Windows") == ["act", "fd", "yq"]
    assert executables(tmp_path, declared, "Linux") == ["gone", "rg"]
