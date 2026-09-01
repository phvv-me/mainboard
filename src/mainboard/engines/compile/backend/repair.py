import csv
from functools import cached_property
from importlib.metadata import distributions
from pathlib import Path
from typing import TYPE_CHECKING

from patos import FrozenOpenModel

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator
    from importlib.metadata import Distribution

# pixi installs its PyPI half through uv, which stamps every distribution it writes with this
# installer, so a conda-owned record belongs to another manager and is never touched here.
_INSTALLER = "uv-pixi"
_ARTIFACT_SUFFIXES = frozenset({".dylib", ".pyd", ".so"})
_SOURCE_NAMES = frozenset({"CMakeLists.txt", "Cargo.toml", "meson.build", "pyproject.toml"})
_SOURCE_SUFFIXES = frozenset({".c", ".cc", ".cpp", ".cu", ".cuh", ".h", ".hpp", ".pyx", ".rs"})
# Directories a build writes into rather than compiles from. Descending into them would let a
# vendored `.venv`, a `target/` of freshly unpacked crates, or a `build/` of copied headers
# date every package as permanently out of date.
_IGNORED_DIRS = frozenset({"__pycache__", "build", "dist", "node_modules", "target"})


class DirInfo(FrozenOpenModel):
    """The `dir_info` half of a PEP 610 record, written when an install came from a directory."""

    editable: bool = False


class DirectUrl(FrozenOpenModel):
    """A PEP 610 `direct_url.json`, saying where an installed distribution came from.

    Only a local editable tree is interesting here, since everything else pixi installs is a
    wheel it can lay down again from the lock alone.
    """

    url: str = ""
    dir_info: DirInfo = DirInfo()

    @property
    def editable(self) -> bool:
        """Whether the distribution imports straight from a source tree somebody still edits."""
        return self.dir_info.editable

    @property
    def source(self) -> Path | None:
        """The local directory an editable was installed from, `None` for anything else."""
        if not self.editable or not self.url.startswith("file://"):
            return None
        return Path.from_uri(self.url)

    @classmethod
    def beside(cls, distribution: Distribution) -> DirectUrl:
        """Parse the record shipped next to ``distribution``, empty when it ships none."""
        return cls.model_validate_json(distribution.read_text("direct_url.json") or "{}")


class InstalledPackage:
    """One uv-installed distribution, judged by what is on disk rather than by what is locked.

    A wheel is judged by its files, since one whose import roots disappeared keeps its
    `dist-info` and still counts as installed. An editable is judged by its clock, since it
    keeps whatever extension was compiled the first time however far its sources have moved on.
    """

    def __init__(self, distribution: Distribution, site_packages: Path) -> None:
        """Bind one distribution to the site-packages tree it was read from.

        distribution: the installed distribution, as `importlib.metadata` found it.
        site_packages: the directory holding its `dist-info` and its import roots.
        """
        self.distribution = distribution
        self.site_packages = site_packages

    @property
    def name(self) -> str:
        """The distribution name pixi would reinstall this package by."""
        return self.distribution.name

    @cached_property
    def origin(self) -> DirectUrl:
        """Where this distribution was installed from."""
        return DirectUrl.beside(self.distribution)

    def artifacts(self) -> list[Path]:
        """The compiled extension modules this install recorded as its own.

        `RECORD` is read here rather than through `Distribution.files`, which silently drops
        every path that has gone missing. A recorded extension nobody can find is exactly the
        state this audit exists to report.
        """
        recorded = self.distribution.read_text("RECORD") or ""
        return [
            self.site_packages / row[0]
            for row in csv.reader(recorded.splitlines())
            if Path(row[0]).suffix in _ARTIFACT_SUFFIXES
        ]

    def damaged(self) -> bool:
        """Whether this wheel declares import roots and not one of them survives.

        A distribution declaring no root claims nothing that could go missing, and an editable
        is left to :meth:`outdated` because it imports through a path hook rather than from
        files under site-packages.
        """
        if self.origin.editable:
            return False
        declared = self.distribution.read_text("top_level.txt") or ""
        roots = [root for line in declared.splitlines() if (root := line.strip())]
        return bool(roots) and not any(self.importable(root) for root in roots)

    def importable(self, root: str) -> bool:
        """Whether ``root`` still resolves to a package directory, a module, or an extension."""
        return (self.site_packages / root).exists() or any(self.site_packages.glob(f"{root}.*"))

    def outdated(self) -> bool:
        """Whether an editable's native sources are newer than the extensions it installed.

        Only a package that compiled something can go out of date this way, so a pure Python
        editable, which imports its sources directly, never comes back true. An extension the
        distribution recorded but that no longer exists dates to zero, which makes every source
        newer and asks for the rebuild that missing file needs.
        """
        source = self.origin.source
        if source is None:
            return False
        artifacts = self.artifacts()
        if not artifacts:
            return False
        built = min((path.stat().st_mtime_ns for path in artifacts if path.exists()), default=0)
        return InstalledPackage._newest_source(source) > built

    @staticmethod
    def _newest_source(tree: Path) -> int:
        """The newest modification time, in nanoseconds, among the files a native build reads.

        Dot directories and build output trees are skipped, and `0` comes back for a tree holding
        nothing a compiler would open, which is every pure Python package.
        """
        newest = 0
        for directory, subdirectories, filenames in tree.walk():
            subdirectories[:] = [
                name
                for name in subdirectories
                if not name.startswith(".") and name not in _IGNORED_DIRS
            ]
            for filename in filenames:
                if filename in _SOURCE_NAMES or Path(filename).suffix in _SOURCE_SUFFIXES:
                    newest = max(newest, (directory / filename).stat().st_mtime_ns)
        return newest


class EnvironmentAudit:
    """Names the PyPI packages an installed environment has to reinstall to be trustworthy.

    `pixi install` reconciles an environment against its lock, which says whether a package is
    recorded as installed and never whether what it left behind still works. Two failures
    survive that. A wheel whose files were removed underneath pixi, by a swapped CUDA provider
    or a half-deleted cache, keeps its `dist-info` and still counts as installed while none of
    its import roots exist. An editable carrying a compiled extension keeps the artifact of its
    first build however far its sources have moved on. Neither is visible in the lock, so the
    audit reads the environment itself and takes a prefix and nothing else.
    """

    def __init__(self, prefix: Path) -> None:
        self.prefix = prefix

    @staticmethod
    def names(packages: Iterable[InstalledPackage]) -> tuple[str, ...]:
        """The distinct distribution names, ordered case-insensitively for a stable argv."""
        return tuple(sorted({package.name for package in packages}, key=str.casefold))

    def damaged(self) -> tuple[str, ...]:
        """The installed wheels whose declared import roots have all disappeared."""
        return self.names(package for package in self.installed() if package.damaged())

    def installed(self) -> Iterator[InstalledPackage]:
        """Every uv-installed distribution across the environment's site-packages trees."""
        candidates = [
            *self.prefix.glob("lib/python*/site-packages"),
            self.prefix / "Lib" / "site-packages",
        ]
        for site_packages in candidates:
            if not site_packages.is_dir():
                continue
            for distribution in distributions(path=[str(site_packages)]):
                if (distribution.read_text("INSTALLER") or "").strip() == _INSTALLER:
                    yield InstalledPackage(distribution, site_packages)

    def suspect(self) -> tuple[str, ...]:
        """Every package to reinstall, the damaged wheels and the editables to rebuild."""
        return self.names(
            package for package in self.installed() if package.damaged() or package.outdated()
        )
