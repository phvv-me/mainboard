# The part of entering an environment that depends on what is installed and where a job landed.
#
# pixi owns activation proper, and on POSIX that stays the shell hook it writes, since conda
# packages and module systems activate themselves in shell. What pixi cannot know is the rest:
# that pip's CUDA wheels keep their shared libraries somewhere the loader never looks, that a
# conda prefix's build search paths are not exported, and that a job which landed on a cluster
# node has fast local scratch its compile caches belong on. Those are read off the prefix and the
# machine every time an environment is entered, here, once, in Python.
#
# Every generated activation calls this module as its last step, `python -m
# mainboard.runtime.activation` printing the lines a shell evaluates, and a runner that enters an
# environment without a shell applies the same `Runtime` in process. A step only ever adds: a
# path list gains what is missing and a variable the caller already set keeps its value.

import os
import shlex
import sys
from abc import ABC, abstractmethod
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping, MutableMapping, Sequence


def prepended(environ: MutableMapping[str, str], name: str, entries: Sequence[Path]) -> None:
    """Put `entries` at the front of the path list `name`, each at most once.

    environ: the environment being built.
    name: the path-list variable, `PATH` or `LD_LIBRARY_PATH` say.
    entries: the directories to lead with, in order.
    """
    current = [entry for entry in environ.get(name, "").split(os.pathsep) if entry]
    leading = [str(entry) for entry in entries if str(entry) not in current]
    if leading:
        environ[name] = os.pathsep.join(dict.fromkeys([*leading, *current]))


def defaulted(environ: MutableMapping[str, str], name: str, value: str) -> None:
    """Set `name` to `value` unless the caller already gave it one, empty counting as none."""
    if not environ.get(name):
        environ[name] = value


class Step(ABC):
    """One fact about the prefix or the machine that entering an environment has to state."""

    @abstractmethod
    def apply(self, environ: MutableMapping[str, str], prefix: Path) -> None:
        """Add what this step knows to `environ`, for an environment installed at `prefix`."""


class WheelLibraries(Step):
    """The shared libraries pip's CUDA wheels install under `site-packages`.

    `nvidia-*` wheels put each library under `nvidia/<name>/lib` and RAPIDS' `lib*` wheels under
    `lib<name>/lib64`, none of which the dynamic loader searches, so a binary that links against
    one outside torch's own `RPATH` fails to load it. Every such directory leads the loader path.
    """

    def apply(self, environ: MutableMapping[str, str], prefix: Path) -> None:
        found = [
            library
            for site in sorted(prefix.glob("lib/python3*/site-packages"))
            for pattern in ("nvidia/*/lib", "lib*/lib64")
            for library in sorted(site.glob(pattern))
            if library.is_dir()
        ]
        prepended(environ, "LD_LIBRARY_PATH", found)


class BuildPaths(Step):
    """The prefix's own `pkg-config` directory and libclang, for native builds inside it.

    A conda prefix installs both and exports neither, so a cargo or meson build that links a
    system library from the environment would otherwise find the host's copy or none at all.
    """

    def apply(self, environ: MutableMapping[str, str], prefix: Path) -> None:
        library = prefix / "lib"
        if (library / "pkgconfig").is_dir():
            prepended(environ, "PKG_CONFIG_PATH", [library / "pkgconfig"])
        if any(library.glob("libclang.so*")):
            defaulted(environ, "LIBCLANG_PATH", str(library))


class CompileCaches(Step):
    """torch.compile's and Triton's caches, on node-local scratch when the job has some.

    A cluster's home and work filesystems are network mounts, and a cold compile against a cache
    on one is the known way an aarch64 compile hangs, so on a node with local scratch both caches
    move there. Scratch counts only on a scheduled node or one with a local mount, never a
    laptop's ordinary temporary directory, and the model caches stay where they are, since the
    weights are large, persistent and would be wiped with the scratch at the end of every job.

    mount: the local mount whose presence marks a node as having node-local scratch.
    """

    candidates = ("LOCALDIR", "PBS_LOCALDIR", "TMPDIR")

    def __init__(self, mount: Path = Path("/local")) -> None:
        self.mount = mount

    def apply(self, environ: MutableMapping[str, str], prefix: Path) -> None:
        del prefix
        scratch = next(
            (
                Path(where)
                for where in (environ.get(name, "") for name in self.candidates)
                if where and Path(where).is_dir() and os.access(where, os.W_OK)
            ),
            None,
        )
        scheduled = environ.get("LOCALDIR") or environ.get("PBS_JOBID") or self.mount.is_dir()
        if scratch is None or not scheduled:
            return
        defaulted(environ, "TORCHINDUCTOR_CACHE_DIR", str(scratch / "torchinductor"))
        defaulted(environ, "TRITON_CACHE_DIR", str(scratch / "triton"))
        # A cache directory that cannot be made is the compiler's to report when it writes
        # there, not a reason the environment cannot be entered at all.
        for name in ("TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR"):
            with suppress(OSError):
                Path(environ[name]).mkdir(parents=True, exist_ok=True)


class Runtime:
    """Every step, applied in order to an environment installed at one prefix.

    prefix: the installed environment, pixi's `CONDA_PREFIX`.
    steps: what entering it states, the house steps unless a caller needs others.
    """

    def __init__(
        self,
        prefix: Path,
        steps: Sequence[Step] = (WheelLibraries(), BuildPaths(), CompileCaches()),
    ) -> None:
        self.prefix = prefix
        self.steps = steps

    def apply(self, environ: MutableMapping[str, str]) -> None:
        """Add every step's facts to `environ` in place."""
        for step in self.steps:
            step.apply(environ, self.prefix)

    def changes(self, environ: Mapping[str, str]) -> dict[str, str]:
        """The variables applying the steps to `environ` would set, and their values."""
        applied = dict(environ)
        self.apply(applied)
        return {name: value for name, value in applied.items() if environ.get(name) != value}

    def shell(self, environ: Mapping[str, str]) -> str:
        """The POSIX shell lines that make `environ` what `apply` would, for a sourced script."""
        return "".join(
            f"export {name}={shlex.quote(value)}\n"
            for name, value in self.changes(environ).items()
        )


def main() -> None:
    """Print the lines finishing the activation the calling shell just performed.

    Reads the prefix the shell entered from `CONDA_PREFIX`, which pixi's hook has just set, and
    prints nothing for a shell that entered none.
    """
    prefix = os.environ.get("CONDA_PREFIX", "")
    if prefix:
        sys.stdout.write(Runtime(Path(prefix)).shell(os.environ))


if __name__ == "__main__":
    main()
