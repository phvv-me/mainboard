# A POSIX machine made to answer the way a bare Windows box does, for the differences that broke
# this package on Windows most: the tools a POSIX shell brings, how a path reads as text, and two
# file-system rules.
#
# THE TOOLS. A Windows center has no bash, sh, rsync, flock or timeout, so code shelling out to
# one works on every Mac and fails only on the Windows runner. The PATH is rebuilt without them
# as a directory of links to everything else, since `/usr/bin` holds git and bash side by side.
# On Windows itself the POSIX tools arrive as whole directories (Git for Windows' `usr/bin`), so
# those are dropped.
#
# THE TEXT. `str(path)` is `pkg/one.py` on macOS and `pkg\one.py` on Windows, and every
# Windows-only failure of that kind was a path turned into text another reader parses (an ssh
# config, a tool's argv, a line an agent reads) and a test expecting forward slashes. So `str()`
# of a path renders with backslashes when first-party code asks, while the interpreter's own
# machinery (`os.fspath`, pathlib itself) keeps the real spelling. Windows' system calls read a
# backslash path as the same file and this machine's do not, so a rendered spelling is read back
# where the system reads it: parsed into a path again, opened, or handed to a process as its
# working directory, its environment or an absolute path in its argv. Only the text differs, as
# on Windows.
#
# THE FILES. A `NamedTemporaryFile` cannot be opened again by name while still open, and a
# symlink needs a privilege an ordinary account lacks. Both are enforced, with Windows' own
# errors, so code that needs a fallback shows whether it has one.

import builtins
import hashlib
import io
import os
import posixpath
import subprocess
import sys
import sysconfig
import tempfile
from functools import cache
from pathlib import Path, PurePath
from typing import IO, TYPE_CHECKING, Concatenate

from .seal import SPAWN

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    import pytest

    from .seal import Command

# Everything `open` accepts as the file: a path in any spelling, or a descriptor.
type OpenTarget = str | bytes | int | os.PathLike[str] | os.PathLike[bytes]

# What a bare Windows machine lacks that a POSIX one takes for granted.
POSIX_TOOLS = frozenset(
    {"bash", "sh", "dash", "zsh", "ksh", "rsync", "flock", "timeout", "setsid", "nohup", "stdbuf"}
)

# Where the interpreter's own code and every installed distribution live, whose frames keep the
# real spelling. Each is taken as sysconfig names it, as its links resolve, and as the loaded
# standard library reports it, because a uv-managed interpreter is reached through a link naming
# another directory than the one its code objects were compiled from.
_DECLARED = [sysconfig.get_path(key) for key in ("stdlib", "platstdlib", "purelib", "platlib")]
_SYSTEM = tuple(
    {
        *_DECLARED,
        *(os.path.realpath(path) for path in _DECLARED),
        os.path.dirname(os.__file__),
        os.path.dirname(posixpath.__file__),
    }
)


@cache
def first_party(filename: str) -> bool:
    """Whether code in `filename` (a `co_filename`) is neither the interpreter's nor installed."""
    return not filename.startswith(("<", *_SYSTEM)) and not os.path.realpath(filename).startswith(
        _SYSTEM
    )


class Spelling:
    """Paths spelled as Windows spells them to first-party code, and read back where they land.

    rendered: every backslash spelling handed out so far, mapped to the real path it stands for.
    """

    def __init__(self) -> None:
        self.rendered: dict[str, str] = {}

    def text(self, spelled: Callable[[PurePath], str]) -> Callable[[PurePath], str]:
        """`PurePath.__str__` as Windows answers it to first-party code, `spelled` elsewhere.

        spelled: the real `__str__`, which pathlib and the interpreter keep reading through.
        """

        def windows(path: PurePath) -> str:
            text = spelled(path)
            caller = sys._getframe(1).f_code.co_filename
            if path.parser is not posixpath or not first_party(caller):
                return text
            backslashed = text.replace("/", "\\")
            self.rendered[backslashed] = text
            return backslashed

        return windows

    def real(self, value: str, *, relative: bool = True) -> str:
        """`value` with every rendered spelling in it read back to the path it stands for.

        relative: read relative spellings back too, not only absolute ones.
        """
        if "\\" not in value:
            return value
        for rendered in sorted(self.rendered, key=len, reverse=True):
            if relative or rendered.startswith("\\"):
                value = value.replace(rendered, self.rendered[rendered])
        return value

    def parsing(self, init: Callable[..., None]) -> Callable[..., None]:
        """`PurePath.__init__` reading a rendered spelling back, as Windows parses its own."""

        def parsed(path: PurePath, *segments: str | os.PathLike[str]) -> None:
            init(path, *(self.real(part) if isinstance(part, str) else part for part in segments))

        return parsed

    def spawning(self, spawn: Callable[..., None]) -> Callable[..., None]:
        """`Popen._execute_child` reading rendered spellings back where the system reads them.

        An absolute spelling in the argv, the working directory and every environment value is
        a path the child opens, which Windows would find. A relative one in the argv stays as
        rendered, since it is also text the child may print back, and a tool echoing `pkg\\one.py`
        is exactly what Windows does.
        """

        def spawned(
            popen: subprocess.Popen[bytes],
            args: Command,
            executable: str | None,
            preexec_fn: Callable[[], None] | None,
            close_fds: bool,
            pass_fds: tuple[int, ...],
            cwd: str | os.PathLike[str] | None,
            env: Mapping[str, str] | None,
            *rest: int | bool | None,
        ) -> None:
            if isinstance(args, str):
                args = self.real(args, relative=False)
            elif not isinstance(args, bytes | os.PathLike):
                args = [
                    self.real(word, relative=False) if isinstance(word, str) else word
                    for word in args
                ]
            if isinstance(cwd, str):
                cwd = self.real(cwd)
            inherited = os.environ if env is None else env
            env = {key: self.real(value) for key, value in inherited.items()}
            spawn(popen, args, executable, preexec_fn, close_fds, pass_fds, cwd, env, *rest)

        return spawned


class Sharing:
    """Windows' rules for the files a test opens: no second open of a live temporary file.

    live: every open `NamedTemporaryFile` that deletes itself, by the name it can be reached at.
    """

    def __init__(self, spelling: Spelling) -> None:
        self.spelling = spelling
        self.live: dict[str, IO[bytes] | IO[str]] = {}

    def temporary[**Made, File: (IO[bytes], IO[str])](
        self, make: Callable[Made, File]
    ) -> Callable[Made, File]:
        """`NamedTemporaryFile`, remembering each self-deleting file while it stays open."""

        def made(*args: Made.args, **named: Made.kwargs) -> File:
            handle = make(*args, **named)
            if named.get("delete", True) and named.get("delete_on_close", True):
                self.live[handle.name] = handle
            return handle

        return made

    def opening[**Rest, Opened](
        self, opener: Callable[Concatenate[OpenTarget, Rest], Opened]
    ) -> Callable[Concatenate[OpenTarget, Rest], Opened]:
        """`open` of a rendered name reaching its file, and refusing a live temporary one."""

        def opened(file: OpenTarget, *args: Rest.args, **named: Rest.kwargs) -> Opened:
            if isinstance(file, str):
                file = self.spelling.real(file)
            held = None if isinstance(file, int) else self.live.get(os.fsdecode(file))
            if held is not None and not held.closed:
                raise PermissionError(
                    13, "The process cannot access the file: another process holds it", file
                )
            return opener(file, *args, **named)

        return opened


def unprivileged(*_: str | os.PathLike[str] | bool) -> None:
    """`os.symlink` as an account without Developer Mode meets it on Windows."""
    raise OSError(1314, "A required privilege is not held by the client")


class WindowsLike:
    """This machine answering a test the way a bare Windows box would.

    farms: the directory the pruned PATHs are built under, one per inherited PATH, kept for the
        session since linking every program on PATH is the slow part.
    platform: the platform building it, `sys.platform` unless a test says otherwise.
    """

    def __init__(self, farms: Path, *, platform: str = sys.platform) -> None:
        self.farms = farms
        self.platform = platform

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Hide the POSIX tools, spell paths and share files as Windows does, while patched."""
        monkeypatch.setenv("PATH", self.path(os.environ["PATH"]))
        spelling = Spelling()
        monkeypatch.setattr(PurePath, "__str__", spelling.text(PurePath.__str__))
        monkeypatch.setattr(PurePath, "__init__", spelling.parsing(PurePath.__init__))
        monkeypatch.setattr(
            subprocess.Popen, SPAWN, spelling.spawning(getattr(subprocess.Popen, SPAWN))
        )
        sharing = Sharing(spelling)
        monkeypatch.setattr(
            tempfile, "NamedTemporaryFile", sharing.temporary(tempfile.NamedTemporaryFile)
        )
        opener = sharing.opening(io.open)
        monkeypatch.setattr(io, "open", opener)
        monkeypatch.setattr(builtins, "open", opener)
        monkeypatch.setattr(os, "symlink", unprivileged)

    def path(self, inherited: str) -> str:
        """The PATH `inherited` with every POSIX tool gone and everything else still found."""
        directories = [Path(entry) for entry in inherited.split(os.pathsep) if entry]
        if self.platform == "win32":
            return os.pathsep.join(
                os.fspath(directory) for directory in directories if not _holds_posix(directory)
            )
        farm = self.farms / hashlib.blake2b(inherited.encode(), digest_size=8).hexdigest()
        if not farm.is_dir():
            self.farms.mkdir(parents=True, exist_ok=True)
            building = Path(tempfile.mkdtemp(dir=self.farms))
            for directory in directories:
                for program in _programs(directory):
                    link = building / program.name
                    if program.name not in POSIX_TOOLS and not link.exists():
                        link.symlink_to(program)
            building.rename(farm)
        return os.fspath(farm)


def _programs(directory: Path) -> list[Path]:
    """The executable files directly in `directory`, none when it is missing or unreadable."""
    try:
        entries = list(directory.iterdir())
    except OSError:
        return []
    return [entry for entry in entries if entry.is_file() and os.access(entry, os.X_OK)]


def _holds_posix(directory: Path) -> bool:
    """Whether `directory` carries a POSIX tool under any of Windows' executable suffixes."""
    return any(
        (directory / f"{tool}{suffix}").is_file()
        for tool in POSIX_TOOLS
        for suffix in ("", ".exe")
    )
