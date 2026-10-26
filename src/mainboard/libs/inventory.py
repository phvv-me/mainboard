import re
from functools import cached_property
from importlib.metadata import distributions
from pathlib import Path

from packaging.utils import canonicalize_name
from patos import FrozenModel, FrozenOpenModel

from ..engines.compile.backend.pixi.repair import site_packages

# A shared library file, with or without the version suffixes the loader's names carry.
_SHARED = re.compile(r"\.so(?:\.\d+)*$")
# The conda packages that carry CUDA libraries: the toolkit's `cuda-*`, `libcu*`, `libnv*`, and
# the standalone cuDNN, NCCL, NVSHMEM and cuTENSOR.
_CONDA_CUDA = re.compile(r"^(?:cuda|cudnn|nccl|nvshmem|cutensor|libcu|libnv)")


class Family(FrozenModel):
    """One CUDA library family, the unit whose members must all be one release.

    prefixes: what the family's sonames begin with, so `libcudnn_graph.so.9` is cuDNN's.
    """

    name: str
    prefixes: tuple[str, ...]

    @property
    def stem(self) -> str:
        """The prefix every soname of the family opens with."""
        return min(self.prefixes, key=len)


_FAMILIES = tuple(
    Family(name=name, prefixes=prefixes)
    for name, prefixes in {
        "cuDNN": ("libcudnn",),
        "cuBLAS": ("libcublas", "libnvblas"),
        "CUDA runtime": ("libcudart",),
        "NCCL": ("libnccl",),
        "nvJitLink": ("libnvJitLink",),
        "NVRTC": ("libnvrtc",),
        "cuFFT": ("libcufft",),
        "cuRAND": ("libcurand",),
        "cuSOLVER": ("libcusolver",),
        "cuSPARSE": ("libcusparse",),
        "cuSPARSELt": ("libcusparseLt",),
        "CUPTI": ("libcupti",),
        "NVTX": ("libnvToolsExt", "libnvtx"),
        "cuFile": ("libcufile",),
        "NVSHMEM": ("libnvshmem",),
        "NVVM": ("libnvvm",),
        "cuTENSOR": ("libcutensor",),
    }.items()
)


def family_of(soname: str) -> Family | None:
    """The family `soname` belongs to, the one with the longest matching prefix."""
    matches = [
        (len(prefix), family)
        for family in _FAMILIES
        for prefix in family.prefixes
        if soname.startswith(prefix)
    ]
    return max(matches, key=lambda match: match[0], default=(0, None))[1]


class Provider(FrozenModel):
    """A package of the environment, as the manifest would name it.

    ecosystem: `python` for a wheel (declared under `[python.deps]`), `conda` otherwise.
    """

    name: str
    version: str
    ecosystem: str


class Provided(FrozenModel):
    """One CUDA library file an environment's package installed."""

    path: Path
    provider: Provider
    family: Family

    @classmethod
    def of(cls, path: Path, provider: Provider) -> Provided | None:
        """`path` as a library of `provider`, None unless it is an existing family member."""
        family = family_of(path.name)
        if family is None or not _SHARED.search(path.name) or not path.exists():
            return None
        return cls(path=path, provider=provider, family=family)


class CondaPackage(FrozenOpenModel):
    """A conda package's `conda-meta` record, reduced to what the inventory reads."""

    name: str
    version: str = ""
    files: tuple[str, ...] = ()


class Inventory:
    """The CUDA libraries an installed environment provides, and which package put each there.

    Read from what the packages recorded (a wheel's RECORD, a conda package's `conda-meta`),
    never from the installer's lock, so it describes the prefix as it is on disk now.

    prefix: the environment's directory.
    """

    @cached_property
    def libraries(self) -> tuple[Provided, ...]:
        """Every family library the environment's wheels and conda packages installed."""
        return (*self._wheels(), *self._conda())

    @cached_property
    def torch(self) -> tuple[Path, ...]:
        """torch's own libraries, which link the families and so decide what a process loads."""
        return tuple(
            library
            for tree in site_packages(self.prefix)
            for library in sorted((tree / "torch" / "lib").glob("lib*.so*"))
            if library.is_file()
        )

    def __init__(self, prefix: Path) -> None:
        self.prefix = prefix

    def _conda(self) -> list[Provided]:
        found: list[Provided | None] = []
        for record in sorted((self.prefix / "conda-meta").glob("*.json")):
            package = CondaPackage.model_validate_json(record.read_bytes())
            if _CONDA_CUDA.match(package.name):
                provider = Provider(name=package.name, version=package.version, ecosystem="conda")
                found += [Provided.of(self.prefix / file, provider) for file in package.files]
        return [library for library in found if library]

    def _wheels(self) -> list[Provided]:
        found: list[Provided | None] = []
        for tree in site_packages(self.prefix):
            for dist in distributions(path=[str(tree)]):
                name = canonicalize_name(dist.name or "")
                if name.startswith("nvidia-"):
                    provider = Provider(name=name, version=dist.version, ecosystem="python")
                    found += [
                        Provided.of(Path(str(dist.locate_file(file))), provider)
                        for file in dist.files or ()
                    ]
        return [library for library in found if library]

    def owner(self, path: Path) -> Provider | None:
        """The package that installed `path`, else the one whose libraries sit beside it.

        A link made after the install (the versioned spelling a library is opened by) has no
        record of its own but is beside the files it points at.
        """
        for provided in self.libraries:
            if provided.path == path:
                return provided.provider
        return next(
            (
                provided.provider
                for provided in self.libraries
                if provided.path.parent == path.parent
            ),
            None,
        )
