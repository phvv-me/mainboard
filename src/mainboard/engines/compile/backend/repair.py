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

# pixi installs its PyPI half through uv, which stamps every distribution it writes with this
# installer, so a conda-owned record belongs to another manager and is never touched here.
_INSTALLER = "uv-pixi"
_ARTIFACT_SUFFIXES = frozenset({".dylib", ".pyd", ".so"})
# The extensions an import resolves through, as opposed to a linked library a `.so` carries.
_EXTENSION_SUFFIXES = frozenset({".pyd", ".so"})
# What a compiler opens. Build configuration is deliberately not here: `pyproject.toml`,
# `CMakeLists.txt` and their kind are rewritten in place by every tool that touches packaging,
# so their clocks move without a single translation unit changing, while a `.cpp` whose clock
# moved is a `.cpp` somebody wrote to.
_SOURCE_SUFFIXES = frozenset({".c", ".cc", ".cpp", ".cu", ".cuh", ".h", ".hpp", ".pyx", ".rs"})
# Directories a build writes into rather than compiles from. Descending into them would let a
# vendored `.venv`, a `target/` of freshly unpacked crates, or a `build/` of copied headers
# date every package as permanently out of date.
_IGNORED_DIRS = frozenset({"__pycache__", "build", "dist", "node_modules", "target"})


def site_packages(prefix: Path) -> tuple[Path, ...]:
    """The site-packages trees a compiled prefix holds, POSIX and Windows spellings alike."""
    candidates = [
        *(prefix / "lib").glob("python*/site-packages"),
        prefix / "Lib" / "site-packages",
    ]
    return tuple(tree for tree in candidates if tree.is_dir())


def recorded_extensions(name: str, *, prefix: Path) -> tuple[Path, ...]:
    """The compiled extensions the environment at `prefix` holds for import name `name`.

    Asked of that environment's own dist-infos and never of the interpreter asking: the process
    dispatching a job is a uv tool whose own site-packages holds none of what the job's target
    environment holds, and it read its own metadata for a day and deferred nothing (2026-09-07).
    The shape read here is the one thing a package's installed form is knowable from without
    running its code: a build backend is free to install a compiled extension anywhere while
    redirecting only the pure Python half of an editable install back to the source, and the
    RECORD is where that split is written down.

    The import name maps to a distribution the way `importlib.metadata.packages_distributions`
    maps it, off a declared `top_level.txt` or inferred from the RECORD's own paths, with the
    same last resort of a distribution named exactly like the import; the first distribution to
    claim the name wins, ordered by distribution name so the answer never depends on directory
    reading order. What comes back is where the environment says the extensions are, whether or
    not every file is still there: a recorded extension gone missing is exactly the state the
    caller has to know about, not one to silently drop.

    name: the top-level import name, the package directory under its import root.
    prefix: the compiled environment's prefix, whose site-packages are read.
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
            return _recorded(dist, site)
    for dist, site in found:
        if str(dist.name or "").casefold() == name.casefold():
            return _recorded(dist, site)
    return ()


def _import_roots(dist: Distribution) -> set[str]:
    """The top-level import names `dist` claims, declared or inferred from its RECORD."""
    declared = (dist.read_text("top_level.txt") or "").split()
    if declared:
        return set(declared)
    record = dist.read_text("RECORD") or ""
    roots: set[str] = set()
    for row in csv.reader(record.splitlines()):
        path = PurePosixPath(row[0])
        if ".." in path.parts or path.parts[0].endswith(".dist-info"):
            continue
        if len(path.parts) > 1:
            roots.add(path.parts[0])
        elif stem := getmodulename(path.name):
            roots.add(stem)
    return roots


def _recorded(dist: Distribution, site: Path) -> tuple[Path, ...]:
    """The compiled extensions `dist`'s RECORD records, absolute, under `site`."""
    record = dist.read_text("RECORD") or ""
    return tuple(
        site / row[0]
        for row in csv.reader(record.splitlines())
        if row and Path(row[0]).suffix in _EXTENSION_SUFFIXES
    )


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
        """Whether an editable's extensions are behind the sources they were compiled from.

        Two ways an extension stops answering for its tree, asked as the two questions they are
        rather than fused into one clock reading. A recorded extension nobody can find is gone,
        and nothing absent can be current. An extension that is there is behind only when a file
        a compiler opens is newer than the newest artifact this install wrote, which is when that
        build finished; measuring against the oldest instead called a package with several
        extensions stale for the very sources its own build was compiling.

        What counts as a source is what a compiler opens, and build configuration is not it.
        `pyproject.toml` and `CMakeLists.txt` are rewritten in place by anything that touches
        packaging, so cutoken read as needing a reinstall on a clock two days ahead of its own
        `.cpp` while being byte-identical to the commit that last changed it (2026-09-05) -- a
        reinstall of an extension newer than everything it was built from.

        Only a package that compiled something can go out of date this way, so a pure Python
        editable, which imports its sources directly, never comes back true.
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
        """The newest modification time, in nanoseconds, among the files a compiler would open.

        Dot directories and build output trees are skipped, and `0` comes back for a tree holding
        nothing to compile, which is every pure Python package.
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
        for tree in site_packages(self.prefix):
            for distribution in distributions(path=[str(tree)]):
                if (distribution.read_text("INSTALLER") or "").strip() == _INSTALLER:
                    yield InstalledPackage(distribution, tree)

    def suspect(self) -> tuple[str, ...]:
        """Every package to reinstall, the damaged wheels and the editables to rebuild."""
        return self.names(
            package for package in self.installed() if package.damaged() or package.outdated()
        )
