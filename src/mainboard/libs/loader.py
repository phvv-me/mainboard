import platform
import shutil
import subprocess  # ruff:ignore[suspicious-subprocess-import]  reason=asks the machine's own ldconfig for its cache, fixed argv only since=2026-10-17
from pathlib import Path
from typing import TYPE_CHECKING

from patos import FrozenModel

from .elf import Elf

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

_ASK_SECONDS = 30
# The directories glibc searches last, after the cache, for a name the cache does not list.
_TRUSTED = ("/lib", "/usr/lib", "/lib64", "/usr/lib64")


def system_cache() -> dict[str, Path]:
    """The soname-to-file map of the machine's `ld.so.cache`, empty where `ldconfig` is absent.

    The first entry of a soname wins, as the loader takes the first.
    """
    ldconfig = shutil.which("ldconfig") or "/sbin/ldconfig"
    try:
        said = subprocess.run(
            [ldconfig, "-p"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=_ASK_SECONDS,
            check=False,
        ).stdout
    except OSError, subprocess.TimeoutExpired:
        return {}
    cache: dict[str, Path] = {}
    for line in said.splitlines():
        name, arrow, target = line.strip().partition(" => ")
        if arrow:
            cache.setdefault(name.split(" ")[0], Path(target))
    return cache


class Found(FrozenModel):
    """Where the loader finds a soname, and by which rule.

    via: the search rule that produced it (`LD_LIBRARY_PATH`, `RUNPATH`, `ld.so.cache`, ...).
    """

    path: Path
    via: str


class Loader:
    """The dynamic loader's search order, replayed over a fixed library path and cache.

    glibc looks for a name in: the requester's `RPATH` (only when it has no `RUNPATH`),
    `LD_LIBRARY_PATH`, the requester's `RUNPATH`, `ld.so.cache`, then the default directories.
    The first existing file wins, which is why a library the environment lacks is taken from the
    host, however well the rest of the environment is arranged.

    library_path: `LD_LIBRARY_PATH` as the activated environment sets it.
    cache: the host's `ld.so.cache` as a soname map.
    trusted: the default directories searched after the cache.
    """

    @staticmethod
    def _trusted_directories() -> tuple[Path, ...]:
        """The default directories of this machine, its multiarch ones included."""
        triplet = f"{platform.machine()}-linux-gnu"
        return tuple(Path(base) / sub for base in _TRUSTED for sub in ("", triplet))

    def __init__(
        self,
        library_path: Sequence[Path],
        cache: Mapping[str, Path],
        trusted: Sequence[Path] | None = None,
    ) -> None:
        self.library_path = tuple(library_path)
        self.cache = cache
        self.trusted = tuple(Loader._trusted_directories() if trusted is None else trusted)

    def find(self, soname: str, requester: Elf) -> Found | None:
        """The file `requester` gets when it asks for `soname`, None when nothing answers."""
        rpath = () if requester.runpath else requester.rpath
        for rule, directories in (
            ("RPATH", rpath),
            ("LD_LIBRARY_PATH", self.library_path),
            ("RUNPATH", requester.runpath),
        ):
            for directory in directories:
                if (directory / soname).exists():
                    return Found(path=directory / soname, via=rule)
        cached = self.cache.get(soname)
        if cached is not None and cached.exists():
            return Found(path=cached, via="ld.so.cache")
        return next(
            (
                Found(path=directory / soname, via="default directories")
                for directory in self.trusted
                if (directory / soname).exists()
            ),
            None,
        )
