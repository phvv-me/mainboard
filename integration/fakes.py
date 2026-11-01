import subprocess
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace

type Option = bool | float | str | Mapping[str, str] | None
type Spawner = Callable[..., subprocess.CompletedProcess[str]]


def disks(free: int) -> Callable[[Path | str], SimpleNamespace]:
    """A stand-in for `shutil.disk_usage` whose every disk has `free` of twice that many bytes."""

    def usage(path: Path | str) -> SimpleNamespace:
        return SimpleNamespace(total=2 * free, used=free, free=free)

    return usage


def exiting(status: int, stdout: str = "") -> Spawner:
    """A stand-in for `subprocess.run` whose child prints `stdout` and exits `status`."""

    def run(argv: Sequence[str], **options: Option) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(argv, status, stdout, "")

    return run


def failing(fault: Exception) -> Spawner:
    """A stand-in for `subprocess.run` that raises `fault`, as a child that cannot answer does."""

    def run(argv: Sequence[str], **options: Option) -> subprocess.CompletedProcess[str]:
        raise fault

    return run
