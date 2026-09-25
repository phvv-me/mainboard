import csv
from functools import cached_property
from importlib.metadata import distributions
from inspect import getmodulename
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

from patos import FrozenOpenModel

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator
    from importlib.metadata import Distribution

# uv stamps every distribution pixi installs with this, so a conda-owned record is never touched.
_INSTALLER = "uv-pixi"
_ARTIFACT_SUFFIXES = frozenset({".dylib", ".pyd", ".so"})
# The extensions an import resolves through, as opposed to a linked library a `.so` carries.
_EXTENSION_SUFFIXES = frozenset({".pyd", ".so"})
# What a compiler opens. Build configuration (`pyproject.toml`, `CMakeLists.txt`) is left out:
# packaging tools rewrite it in place, moving its clock with no translation unit changed.
_SOURCE_SUFFIXES = frozenset({".c", ".cc", ".cpp", ".cu", ".cuh", ".h", ".hpp", ".pyx", ".rs"})
# Directories a build writes into rather than compiles from, which would date every package.
_IGNORED_DIRS = frozenset({"__pycache__", "build", "dist", "node_modules", "target"})


def site_packages(prefix: Path) -> tuple[Path, ...]:
    """The site-packages trees a compiled prefix holds, POSIX and Windows spellings alike."""
    candidates = [
        *(prefix / "lib").glob("python*/site-packages"),
        prefix / "Lib" / "site-packages",
    ]
    return tuple(tree for tree in candidates if tree.is_dir())


def recorded_extensions(name: str, *, prefix: Path) -> tuple[Path, ...]:
    """The compiled extensions the environment at `prefix` records for import name `name`.

    Read from that environment's dist-infos, never the asking interpreter's: the dispatching uv
    tool holds none of the target's packages, and reading its own metadata deferred nothing for
    a day (2026-09-07). The RECORD is where a build backend writes down an editable's split
    between redirected Python sources and extensions installed anywhere.

    The name maps to a distribution as `packages_distributions` maps it (`top_level.txt`, else
    RECORD paths, else a distribution named exactly like it), the first claim winning in
    distribution-name order. Recorded paths come back even when missing, which the caller must
    know about.
    """
    found = sorted(
        (
            (dist, site)
            for site in site_packages(prefix)
            for dist in distributions(path=[str(site)])
        ),
        key=lambda pair: str(pair[0].name or ""),
    )
    for dist, site in found:
        if name in _import_roots(dist):
            return _recorded(dist, site, _EXTENSION_SUFFIXES)
    for dist, site in found:
        if str(dist.name or "").casefold() == name.casefold():
            return _recorded(dist, site, _EXTENSION_SUFFIXES)
    return ()


def _declared_roots(dist: Distribution) -> list[str]:
    """The import roots `dist`'s `top_level.txt` declares."""
    return (dist.read_text("top_level.txt") or "").split()


def _record_paths(dist: Distribution) -> list[str]:
    """The paths `dist`'s RECORD lists, read raw since `Distribution.files` drops missing ones."""
    return [row[0] for row in csv.reader((dist.read_text("RECORD") or "").splitlines()) if row]


def _import_roots(dist: Distribution) -> set[str]:
    """The top-level import names `dist` claims, declared or inferred from its RECORD."""
    if declared := _declared_roots(dist):
        return set(declared)
    roots: set[str] = set()
    for path in map(PurePosixPath, _record_paths(dist)):
        if ".." in path.parts or path.parts[0].endswith(".dist-info"):
            continue
        if len(path.parts) > 1:
            roots.add(path.parts[0])
        elif stem := getmodulename(path.name):
            roots.add(stem)
    return roots


def _recorded(dist: Distribution, site: Path, suffixes: frozenset[str]) -> tuple[Path, ...]:
    """The files with `suffixes` that `dist`'s RECORD lists, absolute under `site`."""
    return tuple(site / path for path in _record_paths(dist) if Path(path).suffix in suffixes)


class DirInfo(FrozenOpenModel):
    """The `dir_info` half of a PEP 610 record, written when an install came from a directory."""

    editable: bool = False


class DirectUrl(FrozenOpenModel):
    """A PEP 610 `direct_url.json`, of interest only for a local editable tree."""

    url: str = ""
    dir_info: DirInfo = DirInfo()

    @property
    def editable(self) -> bool:
        return self.dir_info.editable

    @property
    def source(self) -> Path | None:
        """The local directory an editable was installed from."""
        if not self.editable or not self.url.startswith("file://"):
            return None
        return Path.from_uri(self.url)

    @classmethod
    def beside(cls, distribution: Distribution) -> DirectUrl:
        """Parse the record shipped next to `distribution`, empty when it ships none."""
        return cls.model_validate_json(distribution.read_text("direct_url.json") or "{}")


class InstalledPackage:
    """One uv-installed distribution, judged by what is on disk rather than by what is locked.

    site_packages: the tree holding its `dist-info` and import roots.
    """

    def __init__(self, distribution: Distribution, site_packages: Path) -> None:
        self.distribution = distribution
        self.site_packages = site_packages

    @property
    def name(self) -> str:
        return self.distribution.name

    @cached_property
    def origin(self) -> DirectUrl:
        return DirectUrl.beside(self.distribution)

    def artifacts(self) -> tuple[Path, ...]:
        """The compiled extension modules this install recorded, present or not."""
        return _recorded(self.distribution, self.site_packages, _ARTIFACT_SUFFIXES)

    def damaged(self) -> bool:
        """Whether this wheel declares import roots and not one of them survives.

        Its `dist-info` still counts it installed. An editable is left to `outdated`, since it
        imports through a path hook rather than from site-packages.
        """
        if self.origin.editable:
            return False
        roots = _declared_roots(self.distribution)
        return bool(roots) and not any(self.importable(root) for root in roots)

    def importable(self, root: str) -> bool:
        """Whether `root` still resolves to a package directory, a module, or an extension."""
        return (self.site_packages / root).exists() or any(self.site_packages.glob(f"{root}.*"))

    def outdated(self) -> bool:
        """Whether an editable's extensions are missing or behind the sources they compiled.

        Behind means a compiler input newer than the newest artifact, when the build finished;
        the oldest would call a multi-extension build stale for its own sources. Build config is
        no input: cutoken read stale on a `pyproject.toml` clock two days ahead of its
        byte-identical `.cpp` (2026-09-05). A pure Python editable is never outdated.
        """
        source = self.origin.source
        artifacts = self.artifacts()
        if source is None or not artifacts:
            return False
        if not all(path.exists() for path in artifacts):
            return True
        return InstalledPackage._newest_source(source) > max(
            path.stat().st_mtime_ns for path in artifacts
        )

    @staticmethod
    def _newest_source(tree: Path) -> int:
        """The newest mtime in ns among the files a compiler would open, `0` when there are none.

        Dot directories and build output trees are skipped.
        """
        newest = 0
        for directory, subdirectories, filenames in tree.walk():
            subdirectories[:] = [
                name
                for name in subdirectories
                if not name.startswith(".") and name not in _IGNORED_DIRS
            ]
            for filename in filenames:
                if Path(filename).suffix in _SOURCE_SUFFIXES:
                    newest = max(newest, (directory / filename).stat().st_mtime_ns)
        return newest


class CondaRecord(FrozenOpenModel):
    """A conda package's `conda-meta` record: its name and every file it linked into the prefix."""

    name: str
    files: tuple[str, ...] = ()

    def incomplete(self, prefix: Path) -> bool:
        """Whether a file this package linked into `prefix` is gone.

        Existence alone, which keeps a whole prefix to about a second; a dangling link it made
        still counts as present.
        """
        return not all((prefix / name).exists(follow_symlinks=False) for name in self.files)


class EnvironmentAudit:
    """Names the packages an installed environment has to reinstall to be trustworthy.

    `pixi install` checks a package is recorded, not that it still works. Invisible to the lock:
    a wheel whose files vanished underneath (a swapped CUDA provider, a half-deleted cache), a
    conda package's file something else deleted (cargo uninstalling the `dust` binary it read
    off the package's own `.crates.toml`), and an editable still carrying the extension of its
    first build. So the audit reads the prefix.
    """

    def __init__(self, prefix: Path) -> None:
        self.prefix = prefix

    @staticmethod
    def names(packages: Iterable[InstalledPackage | CondaRecord]) -> tuple[str, ...]:
        """The distinct names, ordered case-insensitively for a stable argv."""
        return tuple(sorted({package.name for package in packages}, key=str.casefold))

    def damaged(self) -> tuple[str, ...]:
        """The wheels with no import root left, and the conda packages missing a file."""
        return self.names(
            [
                *(package for package in self.installed() if package.damaged()),
                *self.incomplete(),
            ]
        )

    def incomplete(self) -> Iterator[CondaRecord]:
        """Every conda package in the prefix missing a file its `conda-meta` record lists."""
        for path in (self.prefix / "conda-meta").glob("*.json"):
            record = CondaRecord.model_validate_json(path.read_bytes())
            if record.incomplete(self.prefix):
                yield record

    def installed(self) -> Iterator[InstalledPackage]:
        """Every uv-installed distribution across the environment's site-packages trees."""
        for tree in site_packages(self.prefix):
            for distribution in distributions(path=[str(tree)]):
                if (distribution.read_text("INSTALLER") or "").strip() == _INSTALLER:
                    yield InstalledPackage(distribution, tree)

    def suspect(self) -> tuple[str, ...]:
        """Every package to reinstall: the damaged ones and the editables to rebuild."""
        return self.names(
            [
                *(
                    package
                    for package in self.installed()
                    if package.damaged() or package.outdated()
                ),
                *self.incomplete(),
            ]
        )
