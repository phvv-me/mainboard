# The part of entering an environment that depends on what is installed and where a job landed.
#
# pixi owns activation proper (on POSIX its shell hook, since conda packages and module systems
# activate in shell). What pixi cannot know is read here off the prefix and the machine on every
# entry: pip's CUDA wheels keep libraries where the loader never looks, a conda prefix's build
# search paths are not exported, and a cluster node has fast local scratch for compile caches.
#
# Every generated activation ends with `python -m mainboard.runtime.activation`, printing lines
# a shell evaluates, and a runner entering without a shell applies the same `Runtime` in process.
# A step only adds: a path list gains what is missing, and a variable already set keeps its value.

import os
import shlex
from abc import ABC, abstractmethod
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING

from ..log import logger

if TYPE_CHECKING:
    from collections.abc import Mapping, MutableMapping, Sequence


def prepended(environ: MutableMapping[str, str], name: str, entries: Sequence[Path]) -> None:
    """Put `entries` at the front of the path list `name` (`PATH` say), each at most once."""
    current = [entry for entry in environ.get(name, "").split(os.pathsep) if entry]
    leading = [str(entry) for entry in entries if str(entry) not in current]
    if leading:
        environ[name] = os.pathsep.join(dict.fromkeys([*leading, *current]))


def _defaulted(environ: MutableMapping[str, str], name: str, *, value: str) -> None:
    """Set `name` to `value` unless the caller already gave it one, empty counting as none."""
    if not environ.get(name):
        environ[name] = value


class Step(ABC):
    """One fact about the prefix or the machine that entering an environment has to state."""

    @abstractmethod
    def apply(self, environ: MutableMapping[str, str], prefix: Path) -> None:
        """Add what this step knows to `environ`, for an environment installed at `prefix`."""


class WheelLibraries(Step):
    """The shared libraries pip's CUDA wheels install under `site-packages`, leading the loader.

    `nvidia-*` wheels use `nvidia/<name>/lib` and RAPIDS' `lib*` wheels `lib<name>/lib64`, which
    the loader never searches, so a binary linking one outside torch's `RPATH` fails to load it.
    """

    @staticmethod
    def _libraries(site: Path) -> list[Path]:
        """The wheel library directories under `site`, each pattern's matches sorted in turn."""
        return [
            library
            for pattern in ("nvidia/*/lib", "lib*/lib64")
            for library in sorted(site.glob(pattern))
            if library.is_dir()
        ]

    def apply(self, environ: MutableMapping[str, str], prefix: Path) -> None:
        sites = sorted(prefix.glob("lib/python3*/site-packages"))
        found = [library for site in sites for library in self._libraries(site)]
        prepended(environ, "LD_LIBRARY_PATH", found)


class CudnnSublibraries(Step):
    """The cuDNN wheel's sublibraries under the full versioned names cuDNN 9.2x dlopens.

    The wheel ships `libcudnn_graph.so.9` only, while cuDNN asks for `libcudnn_graph.so.9.27.0`.
    Found nowhere on `LD_LIBRARY_PATH`, that name falls through to a host's own cuDNN of the same
    version, a CUDA 12 build that aborts on `libcublasLt.so.12`. A link beside each library makes
    the environment's own win, in every environment whatever its activation scripts.
    """

    @staticmethod
    def _link_versioned(dist: Path) -> None:
        """Link each cuDNN library beside the wheel's `dist` info under the version it names."""
        version = dist.stem.partition("-")[2].split(".")
        minor_patch = ".".join(version[1:3])
        for library in (dist.parent / "nvidia/cudnn/lib").glob("libcudnn*.so.9"):
            with suppress(OSError):
                library.with_name(f"{library.name}.{minor_patch}").symlink_to(library.name)

    def apply(self, environ: MutableMapping[str, str], prefix: Path) -> None:
        del environ
        for site in prefix.glob("lib/python3*/site-packages"):
            for dist in site.glob("nvidia_cudnn_cu13-*.dist-info"):
                self._link_versioned(dist)


class BuildPaths(Step):
    """The prefix's own `pkg-config` directory and libclang, for native builds inside it.

    A conda prefix installs both and exports neither, so a cargo or meson build would find the
    host's copy or none.
    """

    def apply(self, environ: MutableMapping[str, str], prefix: Path) -> None:
        library = prefix / "lib"
        if (library / "pkgconfig").is_dir():
            prepended(environ, "PKG_CONFIG_PATH", [library / "pkgconfig"])
        if any(library.glob("libclang.so*")):
            _defaulted(environ, "LIBCLANG_PATH", value=str(library))


class CompileCaches(Step):
    """torch.compile's and Triton's caches, on node-local scratch when the job has some.

    A cold compile against a cache on a cluster's network mounts is the known way an aarch64
    compile hangs. Scratch counts only on a scheduled node or one with `mount`, never a laptop's
    temporary directory. Model caches stay put: the weights are large, persistent, and would be
    wiped with the scratch after every job.
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
        _defaulted(environ, "TORCHINDUCTOR_CACHE_DIR", value=str(scratch / "torchinductor"))
        _defaulted(environ, "TRITON_CACHE_DIR", value=str(scratch / "triton"))
        # A cache directory that cannot be made is the compiler's to report, not a refusal here.
        for name in ("TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR"):
            with suppress(OSError):
                Path(environ[name]).mkdir(parents=True, exist_ok=True)


class Runtime:
    """Every step, applied in order to the environment installed at `prefix` (`CONDA_PREFIX`)."""

    def __init__(
        self,
        prefix: Path,
        steps: Sequence[Step] = (
            WheelLibraries(),
            CudnnSublibraries(),
            BuildPaths(),
            CompileCaches(),
        ),
    ) -> None:
        self.prefix = prefix
        self.steps = steps

    def apply(self, environ: MutableMapping[str, str]) -> None:
        for step in self.steps:
            step.apply(environ, self.prefix)

    def changes(self, environ: Mapping[str, str]) -> dict[str, str]:
        """The variables applying the steps to `environ` would set."""
        applied = dict(environ)
        self.apply(applied)
        return {name: value for name, value in applied.items() if environ.get(name) != value}

    def entered(self, environ: Mapping[str, str]) -> dict[str, str]:
        """`environ` as a job enters the environment, its `lib/` leading the loader path.

        Every step is applied, then the prefix's own `lib/` leads, so a library the environment
        ships wins over the host's copy and a wheel's.
        """
        entered = dict(environ)
        self.apply(entered)
        prepended(entered, "LD_LIBRARY_PATH", [self.prefix / "lib"])
        return entered

    def shell(self, environ: Mapping[str, str]) -> str:
        """The POSIX shell lines making `environ` what `apply` would."""
        return "".join(
            f"export {name}={shlex.quote(value)}\n"
            for name, value in self.changes(environ).items()
        )


def _main() -> None:
    """Print the lines finishing the activation of the `CONDA_PREFIX` pixi's hook just set."""
    prefix = os.environ.get("CONDA_PREFIX", "")
    if prefix:
        logger.result(Runtime(Path(prefix)).shell(os.environ), end="")


if __name__ == "__main__":
    _main()
