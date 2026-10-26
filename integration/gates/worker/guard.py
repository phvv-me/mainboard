"""Python diagnostics for a gate worker; the OS sandbox is the boundary, never this hook."""

import fcntl
import os
import sys
from collections.abc import Sequence
from pathlib import Path

type Argument = str | bytes | Path | int | tuple | None

_WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
_LOCAL_NAMES = {"localhost", "127.0.0.1", "::1", None}
_PROCESSES = ["subprocess.Popen", "os.exec", "os.posix_spawn", "os.system"]

# The paths each event writes: (path argument, dir_fd argument or None, follow a final symlink).
_WRITES: dict[str, list[tuple[int, int | None, bool]]] = {
    "os.remove": [(0, 1, False)],
    "os.rmdir": [(0, 1, False)],
    "os.rename": [(0, 2, False), (1, 3, False)],
    "os.link": [(0, 2, True), (1, 3, True)],
    "os.symlink": [(1, 2, False)],
    "os.chmod": [(0, 2, True)],
    "os.chown": [(0, 3, True)],
    "os.utime": [(0, 3, True)],
    "os.truncate": [(0, None, True)],
}


class Guard:
    """Refuses writes outside `roots` and network names, and children when `deny_processes`."""

    @classmethod
    def from_environment(cls) -> Guard:
        return cls([Path(os.environ[key]) for key in ("MB_GATE_ROOT", "MB_GATE_REPORT")])

    @staticmethod
    def _where(descriptor: int) -> str:
        if sys.platform == "darwin":
            found = fcntl.fcntl(descriptor, fcntl.F_GETPATH, bytes(1024))
            return os.fsdecode(found.split(b"\0", 1)[0])
        return os.readlink(f"/proc/self/fd/{descriptor}")

    @staticmethod
    def descriptor(descriptor: int) -> Path | None:
        """Where an open descriptor lives; None for a pipe or socket, which names no path."""
        try:
            where = Guard._where(descriptor)
        except OSError:
            return None
        return Path(where) if where.startswith("/") else None

    def __init__(self, roots: Sequence[Path]) -> None:
        self.roots = [path.resolve() for path in roots]
        self.deny_processes = False
        self.checks = {
            "open": self._opened,
            "os.mkdir": self._made,
            "socket.bind": self._addressed,
            "socket.connect": self._addressed,
            "socket.getaddrinfo": self._resolved,
            **dict.fromkeys(_PROCESSES, self._spawned),
        }

    def _addressed(self, args: Sequence[Argument]) -> None:
        address = args[1]
        if isinstance(address, str):
            self.writable(address)
        elif os.environ.get("MB_GATE_OS") != "linux":
            raise PermissionError(f"M00 denies network endpoint: {address!r}")

    def _made(self, args: Sequence[Argument]) -> None:
        """mkdir(exist_ok=True) on a directory already present changes nothing."""
        match args:
            case (str() | bytes() | Path() as path, _, int() as directory) if not self.path(
                path, directory
            ).is_dir():
                self.writable(path, directory)

    def _opened(self, args: Sequence[Argument]) -> None:
        """Wrapping a descriptor creates no file, so only a path opened for writing is checked."""
        match args:
            case (str() | bytes() | Path() as path, _, int() as flags) if flags & _WRITE_FLAGS:
                self.writable(path)

    def _resolved(self, args: Sequence[Argument]) -> None:
        if os.environ.get("MB_GATE_OS") != "linux" or args[0] not in _LOCAL_NAMES:
            raise PermissionError(f"M00 denies DNS: {args[0]!r}")

    def _spawned(self, args: Sequence[Argument]) -> None:
        if self.deny_processes:
            raise PermissionError("M00 inspection refuses subprocess execution")

    def audit(self, event: str, args: Sequence[Argument]) -> None:
        """Raise PermissionError for a forbidden `event`, before the system is asked."""
        for path, directory, follow_leaf in _WRITES.get(event, []):
            match args[path], -1 if directory is None else args[directory]:
                case (str() | bytes() | Path() | int() as target, int() as anchor):
                    self.writable(target, anchor, follow_leaf=follow_leaf)
                case found:
                    raise PermissionError(f"M00 cannot read where {event} writes: {found!r}")
        if check := self.checks.get(event):
            check(args)

    def install(self) -> None:
        sys.addaudithook(self.audit)

    def path(
        self, value: str | bytes | Path, directory: int = -1, *, follow_leaf: bool = True
    ) -> Path:
        """The absolute, symlink-free place `value` names, relative to `directory` if open."""
        path = Path(os.fsdecode(value))
        if not path.is_absolute() and directory >= 0:
            match self.descriptor(directory):
                case Path() as base:
                    path = base / path
                case None:
                    raise PermissionError(f"M00 cannot place {path} under descriptor {directory}")
        return path.resolve() if follow_leaf else path.parent.resolve() / path.name

    def writable(
        self, value: str | bytes | Path | int, directory: int = -1, *, follow_leaf: bool = True
    ) -> None:
        path = (
            self.descriptor(value)
            if isinstance(value, int)
            else self.path(value, directory, follow_leaf=follow_leaf)
        )
        if path is None or path == Path(os.devnull):
            return
        if any(path.is_relative_to(root) for root in self.roots):
            return
        raise PermissionError(f"M00 denies write outside temporary/report roots: {path}")
