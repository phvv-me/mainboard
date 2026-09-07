# The closure of a job: exactly the files it needs, found by reading them and never by running
# them. A job file imports the GPU libraries of the environment it runs in, and the machine
# dispatching it may hold none, so the walk is over syntax alone.
#
# WHAT SHIPS. The job's own directory, the node, in full, since a node keeps its registration and
# its notes beside its code. Every first-party module the job imports, transitively, from the
# node's own import root and from the roots the manifest installs editable. The workspace
# manifest, which declares the environment the job activates. Nothing else: a job that never
# names a package never carries it, and an edit to that package never marks the job's receipts.
#
# A DISTRIBUTION SHIPS WHOLE. A module found under an import root the manifest installs ships
# with its whole top-level package, not module by module. Two of the three house packages a
# node imports resolve names at run time (`mainboard`'s lazy facade, `reproducibility.fp`'s
# `__getattr__`), packages carry data files their code opens by `importlib.resources`, and half a
# package ahead of the environment's whole one on `PYTHONPATH` shadows the half it left behind.
# A distribution is the unit its own `pyproject.toml` declares and its editable install exposes,
# so it is the unit that ships. The node's own tree is explicit code and ships module by module.
#
# NEEDS ARE NOT CODE. A path the job declares it reads is reached inside the snapshot by a link
# back to the mirror: never copied, never digested, and never allowed to sit over a shipped file.
#
# A COMPILED EXTENSION IS NOT SOURCE. The walk reads a package's `.py` files and an editable
# install redirects exactly those to the tree; a compiled `_native.so` is neither, and no `.py`
# beside it says where the built bytes live. `importlib.metadata` is asked instead, the one
# place a package's installed shape is knowable without running it (`packages_distributions`
# plus `distribution(...).files`, the same reading `trials.provenance.installed` already
# trusts): the answer for `cutoken` is that nanobind's `_native` is not under the source tree at
# all, but physically installed beside a scikit-build-core editable redirect, in the
# environment's own `site-packages` (2026-09-07). An extension found inside the package's own
# directory ships beside it, `built` in the listing since git may hold no opinion on a generated
# file at all. An extension found outside the tree defers the whole package to the environment
# instead of shipping half of it: a stale pure-Python half under this closure's digest beside a
# live compiled half resolved from wherever the host's editable install happens to point is not
# a closure, so nothing of that package ships and the runner's finder lets every import of it
# through unchecked, trusting the same environment install the job's `PYTHONPATH` would otherwise
# have shadowed.

import ast
from importlib.metadata import PackageNotFoundError, distribution, packages_distributions
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

from patos import FrozenModel

from ..core.errors import MissionError
from ..core.project import Project
from ..dispatch.provenance import Repositories, Repository
from .target import Target, dotted, home_of

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

# The dynamic import spellings whose literal argument the walk still follows.
_DYNAMIC = ("import_module", "__import__")


class Module(FrozenModel):
    """One first-party module the walk reached, and the import root it was found under.

    path: the module file, workspace-relative, as the snapshot spells it.
    root: the import root it resolves from, workspace-relative.
    """

    path: str
    root: str


class Walker:
    """The static import walk over a job's first-party roots.

    root: the workspace root every spelled path is relative to.
    home: the job's own import root, workspace-relative, searched first.
    distributions: the roots the manifest installs editable, workspace-relative, in order.
    """

    def __init__(self, root: Path, *, home: str, distributions: Sequence[str]) -> None:
        self.root = root
        self.home = home
        self.roots = list(dict.fromkeys([home, *distributions]))

    def reach(self, file: str) -> list[Module]:
        """Every first-party module `file` imports, transitively, `file` itself first."""
        first = Module(path=file, root=self.home)
        reached: dict[str, Module] = {file: first}
        pending = [first]
        while pending:
            module = pending.pop()
            for name in self.imported(module):
                for found in self.resolve(name):
                    if found.path not in reached:
                        reached[found.path] = found
                        pending.append(found)
        return list(reached.values())

    def imported(self, module: Module) -> Iterator[str]:
        """Every module name `module` imports, relative ones spelled out, in syntax order."""
        tree = ast.parse((self.root / module.path).read_text(encoding="utf-8"))
        package = dotted(self.root / module.path, home=self.root / module.root)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                yield from (alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                base = _absolute(package, level=node.level, name=node.module or "")
                if base is None:
                    continue
                yield base
                yield from (f"{base}.{alias.name}" for alias in node.names if alias.name != "*")
            elif isinstance(node, ast.Call) and (literal := _dynamic(node)):
                yield literal

    def resolve(self, name: str) -> list[Module]:
        """The files `name` imports, parent packages included, under the one root holding it.

        Two roots holding the same top-level package is refused rather than settled by order:
        two campaigns each keep an `experiments` package, and a job that silently imported the
        other one's would run code its receipts never named. A namespace portion is walked
        through wherever it is, since portions are what several roots legitimately share.
        """
        parts = name.split(".")
        holders = [root for root in self.roots if self.__regular(self.root / root / parts[0])]
        if len(holders) > 1:
            raise MissionError(
                f"`{parts[0]}` is a package under more than one import root "
                f"({', '.join(holders)}); a job's imports have to name one, so keep the package "
                "in one root or import it by the root that owns it"
            )
        for root in self.roots:
            chain = self.__chain(self.root / root, parts)
            if chain is not None:
                return [
                    Module(path=file.relative_to(self.root).as_posix(), root=root)
                    for file in chain
                ]
        return []

    @staticmethod
    def __regular(package: Path) -> bool:
        """Whether `package` is a regular package or a module, the shapes one root must own."""
        return (package / "__init__.py").is_file() or package.with_suffix(".py").is_file()

    def __chain(self, base: Path, parts: Sequence[str]) -> list[Path] | None:
        """The package files `parts` walks through under `base`, None when the chain breaks."""
        found: list[Path] = []
        for depth in range(1, len(parts) + 1):
            package = base.joinpath(*parts[:depth])
            if (package / "__init__.py").is_file():
                found.append(package / "__init__.py")
            elif depth == len(parts) and package.with_suffix(".py").is_file():
                found.append(package.with_suffix(".py"))
            elif not package.is_dir():
                return None
        return found


class Closure(FrozenModel):
    """Everything one job ships, computed without importing it.

    target: the job, as spelled.
    owner: the repository holding the job file, whose HEAD names the job's provenance.
    files: every shipped file, workspace-relative, sorted.
    roots: the import roots the job's `PYTHONPATH` names, the node's own first.
    first_party: every top-level name the workspace's import roots define, shipped or not, so
        the runner can refuse one the closure left out instead of reading it from the mirror.
    needs: workspace-relative paths the job reads on the host, linked back to the mirror.
    fetch: the results path the job declared, empty when it declared none.
    built: workspace-relative paths of compiled extensions shipped beside their package's
        source, marked `built` in the listing regardless of what git makes of them.
    deferred: top-level names whose whole distribution the closure left to the environment
        because one of its compiled extensions resolves outside the tree; the runner's finder
        admits every import of these rather than checking them against the listing.
    """

    target: Target
    owner: Repository | None
    files: tuple[str, ...]
    roots: tuple[str, ...]
    first_party: tuple[str, ...] = ()
    needs: tuple[str, ...] = ()
    fetch: str = ""
    built: tuple[str, ...] = ()
    deferred: tuple[str, ...] = ()

    @classmethod
    def of(
        cls,
        target: Target,
        *,
        root: Path,
        distributions: Sequence[str],
        needs: Sequence[str] = (),
    ) -> Closure:
        """The closure of `target`: the node in full, what it imports, and what it declared.

        target: the job.
        root: the workspace root.
        distributions: the import roots the manifest installs editable, workspace-relative.
        needs: data paths declared at dispatch time, joining the ones the file declares.
        """
        repositories = Repositories(root)
        home = home_of(root / target.file, root=root).relative_to(root).as_posix()
        walker = Walker(root, home=home, distributions=distributions)
        reached = walker.reach(target.file)
        declared = target.declaration(root)
        files = set(repositories.kept(target.node))
        built: set[str] = set()
        deferred: set[str] = set()
        decided: dict[str, tuple[bool, tuple[str, ...]]] = {}
        for module in reached:
            if module.root not in distributions:
                files.add(module.path)
                continue
            package = cls.__package(module)
            if package not in decided:
                decided[package] = cls.__compiled(package, root=root)
            outside, extensions = decided[package]
            if outside:
                deferred.add(PurePosixPath(package).name)
            else:
                files.update(repositories.kept(package))
                built.update(extensions)
        files.update(built)
        for resource in declared.resources:
            files.update(cls.__pinned(resource, root, repositories))
        files.add(Project().manifest)
        wanted = tuple(dict.fromkeys([*declared.needs, *needs]))
        cls.__admissible(wanted, files)
        return cls(
            target=target,
            owner=repositories.owning(root / target.node),
            files=tuple(sorted(files)),
            roots=tuple(
                place
                for place in walker.roots
                if any(file == place or file.startswith(f"{place}/") for file in files)
            ),
            first_party=tuple(
                sorted({name for place in walker.roots for name in _defined(root / place)})
            ),
            needs=wanted,
            fetch=declared.fetch,
            built=tuple(sorted(built)),
            deferred=tuple(sorted(deferred)),
        )

    @staticmethod
    def __pinned(resource: str, root: Path, repositories: Repositories) -> list[str]:
        """The files a declared resource pins: itself, or everything kept under it."""
        posix = PurePosixPath(resource)
        if posix.is_absolute() or ".." in posix.parts:
            raise MissionError(f"a resource must be a workspace-relative path, not {resource!r}")
        if (root / resource).is_dir():
            return repositories.kept(resource)
        if (root / resource).is_file():
            return [posix.as_posix()]
        raise MissionError(f"the declared resource {resource!r} is not in the workspace")

    @staticmethod
    def __package(module: Module) -> str:
        """The top-level package directory `module` belongs to under its distribution root."""
        inside = PurePosixPath(module.path).relative_to(module.root)
        return (PurePosixPath(module.root) / inside.parts[0]).as_posix()

    @staticmethod
    def __compiled(package: str, *, root: Path) -> tuple[bool, tuple[str, ...]]:
        """Whether `package` defers to the environment, and which of its extensions ship inside.

        package: the top-level package directory, workspace-relative.
        root: the workspace root.
        """
        tree = (root / package).resolve()
        inside: list[str] = []
        for file in _extension_files(PurePosixPath(package).name):
            try:
                relative = file.resolve().relative_to(tree)
            except ValueError:
                return True, ()
            inside.append((PurePosixPath(package) / relative).as_posix())
        return False, tuple(inside)

    @staticmethod
    def __admissible(needs: Sequence[str], files: set[str]) -> None:
        """Refuse a need that could not be linked: one leaving the workspace, or one over code."""
        for need in needs:
            posix = PurePosixPath(need)
            if posix.is_absolute() or ".." in posix.parts:
                raise MissionError(f"a need must be a workspace-relative path, not {need!r}")
            shadowed = [file for file in files if file == need or file.startswith(f"{need}/")]
            if shadowed:
                raise MissionError(
                    f"the need {need!r} would sit over shipped code ({shadowed[0]}); a need is "
                    "data the job reads, declared beside the code rather than around it"
                )


def _extension_files(name: str) -> tuple[Path, ...]:
    """The compiled extensions an installed distribution named `name` carries, absolute.

    Read off `importlib.metadata` rather than the tree, since a build backend is free to
    install a compiled extension anywhere while redirecting only the pure Python half of an
    editable install back to the source: the metadata is the one place that installed shape is
    knowable without running the distribution's own code.
    """
    for candidate in packages_distributions().get(name, (name,)):
        try:
            found = distribution(candidate)
        except PackageNotFoundError:
            continue
        return tuple(
            Path(file.locate()) for file in found.files or () if file.suffix in (".so", ".pyd")
        )
    return ()


def _absolute(package: str, *, level: int, name: str) -> str | None:
    """`name` as imported from `package` with `level` leading dots, None past the top package.

    package: the importing module's dotted name.
    level: how many leading dots the import carries, zero for an absolute one.
    name: the module named after the dots, empty for `from . import x`.
    """
    if not level:
        return name
    parts = package.split(".")[:-1]
    if level - 1 >= len(parts):
        return None
    base = parts[: len(parts) - (level - 1)]
    return ".".join([*base, name] if name else base)


def _dynamic(call: ast.Call) -> str:
    """The literal module name `call` imports at run time, empty for any other call.

    The one dynamic form followed: `import_module("a.b")` or `__import__("a.b")` with the name
    spelled out. A relative literal or a computed one says nothing a walk can read.
    """
    func = call.func
    named = func.id if isinstance(func, ast.Name) else getattr(func, "attr", "")
    if named not in _DYNAMIC or not call.args:
        return ""
    literal = call.args[0]
    if not isinstance(literal, ast.Constant) or not isinstance(literal.value, str):
        return ""
    return "" if literal.value.startswith(".") else literal.value


def _defined(root: Path) -> set[str]:
    """The top-level names importable from `root`: its packages and its bare modules."""
    try:
        entries = list(root.iterdir())
    except OSError:
        return set()
    return {entry.name for entry in entries if (entry / "__init__.py").is_file()} | {
        entry.stem for entry in entries if entry.suffix == ".py" and entry.is_file()
    }
