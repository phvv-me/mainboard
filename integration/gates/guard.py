# Diagnostic Python boundary; native code is contained by the OS worker, never by this hook.
import fcntl
import os
import sys
from pathlib import Path


class Guard:
    def __init__(self, roots: tuple[Path, ...]) -> None:
        self.roots = tuple(path.resolve() for path in roots)
        self.deny_processes = False

    @classmethod
    def from_environment(cls) -> Guard:
        return cls(tuple(Path(os.environ[key]) for key in ("MB_GATE_ROOT", "MB_GATE_REPORT")))

    def path(
        self, value: str | bytes | Path | int, directory: int = -1, *, follow_leaf: bool = True
    ) -> Path:
        if isinstance(value, int):
            directory, value = value, "."
        path = Path(os.fsdecode(value))
        if not path.is_absolute() and directory not in {-1, None}:
            if sys.platform == "darwin":
                parent = fcntl.fcntl(directory, 50, bytes(1024)).split(b"\0", 1)[0]
                path = Path(os.fsdecode(parent)) / path
            else:
                path = Path(f"/proc/self/fd/{directory}").resolve() / path
        return path.resolve() if follow_leaf else path.parent.resolve() / path.name

    def writable(
        self, value: str | bytes | Path | int, directory: int = -1, *, follow_leaf: bool = True
    ) -> None:
        path = self.path(value, directory, follow_leaf=follow_leaf)
        if path == Path(os.devnull) or any(path.is_relative_to(root) for root in self.roots):
            return
        raise PermissionError(f"M00 denies write outside temporary/report roots: {path}")

    def audit(self, event: str, args: tuple) -> None:
        if event == "open" and args[2] & (
            os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
        ):
            self.writable(args[0])
        elif event == "os.mkdir":
            # mkdir(exist_ok=True) on a directory already present does not change it.
            if not self.path(args[0], args[2]).is_dir():
                self.writable(args[0], args[2])
        elif event in {"os.remove", "os.rmdir"}:
            self.writable(args[0], args[1], follow_leaf=False)
        elif event == "os.rename":
            self.writable(args[0], args[2], follow_leaf=False)
            self.writable(args[1], args[3], follow_leaf=False)
        elif event == "os.link":
            self.writable(args[0], args[2])
            self.writable(args[1], args[3])
        elif event == "os.symlink":
            self.writable(args[1], args[2], follow_leaf=False)
        elif event == "os.chmod":
            self.writable(args[0], args[2])
        elif event in {"os.chown", "os.utime"}:
            self.writable(args[0], args[3])
        elif event == "os.truncate":
            self.writable(args[0])
        elif event in {"socket.connect", "socket.bind"}:
            address = args[1]
            if isinstance(address, str):
                self.writable(address)
            elif os.environ.get("MB_GATE_OS") != "linux":
                raise PermissionError(f"M00 denies network endpoint: {address}")
        elif event == "socket.getaddrinfo":
            if os.environ.get("MB_GATE_OS") != "linux" or args[0] not in {
                "localhost", "127.0.0.1", "::1", None,
            }:
                raise PermissionError(f"M00 denies DNS: {args[0]}")
        elif event in {"subprocess.Popen", "os.exec", "os.posix_spawn", "os.system"}:
            if self.deny_processes:
                raise PermissionError("M00 inspection refuses subprocess execution")
            executable = Path(os.fsdecode(args[0])).name
            if executable in {"ssh", "scp", "sftp", "rsync", "curl", "wget", "vastai"}:
                raise PermissionError(f"M00 denies real transport: {executable}")

    def install(self) -> None:
        sys.addaudithook(self.audit)
