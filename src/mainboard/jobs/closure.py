# The closure of a job: exactly the files it needs, found by reading them and never by running
# them, since a job file imports GPU libraries the dispatching machine may not hold.
#
# WHAT SHIPS. The job's directory, the node, in full (registration and notes live beside code);
# every first-party module it imports, transitively, from the node's import root and the roots the
# manifest installs editable; and the workspace manifest. Nothing else, so an edit to a package a
# job never names never marks its receipts.
#
# A DISTRIBUTION SHIPS WHOLE. A module under an editable root ships with its whole top-level
# package: house packages resolve names at run time (`mainboard`'s lazy facade,
# `reproducibility.fp`'s `__getattr__`), packages open data files by `importlib.resources`, and
# half a package ahead of the environment's whole one on `PYTHONPATH` shadows the other half. The
# node's own tree is explicit code and ships module by module.
#
# NEEDS ARE NOT CODE. A declared data path is linked back to the mirror inside the snapshot: never
# copied, never digested, never allowed over a shipped file.
#
# A COMPILED EXTENSION IS NOT SOURCE. No `.py` says where a built `_native.so` lives, so the job's
# target environment is asked (its site-packages' dist-info RECORDs, via
# `engines.compile.backend.repair`), never the asking interpreter: the dispatcher is a uv tool
# holding none of the job's packages, and reading its own metadata deferred nothing for a day
# (2026-09-07). `cutoken`'s nanobind `_native` sits beside a scikit-build-core editable redirect
# in the environment's site-packages, outside the tree. An extension inside the package directory
# ships beside it, marked `built` since git may hold no opinion on a generated file. One outside
# defers the whole package to the environment: a stale pure-Python half beside a live compiled
# half from wherever the host's install points is not a closure, so none of it ships and the
# runner's finder admits every import of it.
#
# A PYTEST TARGET'S HARNESS IS NOT ITS IMPORTS. Fixtures live in conftest.py files nothing
# imports, so the walk takes every ancestor conftest with its imports, plus what a literal
# `pytest_plugins` names (the one spelling readable without importing). The nearest pytest
# configuration above the node ships too: `pytest.ini` by name, the others by the section they
# carry.

import ast
import tomllib
from configparser import ConfigParser
from functools import partial
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

from patos import FrozenModel

from ..core.errors import MissionError
from ..core.project import Project
from ..dispatch.provenance import SourceTree
from ..engines.compile.backend.repair import recorded_extensions
from .pins import Pin, split
from .target import Target, dotted, home_of, parsed

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

# The dynamic import spellings whose literal argument the walk still follows.
_DYNAMIC = ("import_module", "__import__")


class Module(FrozenModel):
    """One first-party module file the walk reached, and its import root, workspace-relative."""

    path: str
    root: str


class Walker:
    """The static import walk over a job's first-party roots.

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
        # `_absolute` drops the importing module's last component, but an initializer imports
        # relative to itself, so it keeps one to drop.
        if Path(module.path).name == "__init__.py":
            package = f"{package}.__init__"
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

        Two roots holding one top-level package is refused rather than settled by order (two
        campaigns each keep an `experiments` package, and importing the other's would run code
        the receipts never named). Namespace portions, which roots legitimately share, are walked
        wherever they are.
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

    files: every shipped file, workspace-relative, sorted.
    roots: the import roots the job's `PYTHONPATH` names, the node's own first.
    first_party: every top-level name the workspace's import roots define, shipped or not, so
        the runner can refuse one the closure left out instead of reading it from the mirror.
    needs: workspace-relative paths the job reads on the host, linked back to the mirror.
    pins: `hf://org/name@revision/filename` Hub files staged from this machine's cache.
    fetch: the results path the job declared, empty when it declared none.
    built: compiled extensions shipped beside their package's source.
    deferred: top-level names whose whole distribution was left to the environment.
    """

    target: Target
    files: tuple[str, ...]
    roots: tuple[str, ...]
    first_party: tuple[str, ...] = ()
    needs: tuple[str, ...] = ()
    pins: tuple[str, ...] = ()
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
        environment: Path,
        needs: Sequence[str] = (),
    ) -> Closure:
        """The closure of `target`: the node in full, what it imports, and what it declared.

        distributions: the import roots the manifest installs editable, workspace-relative.
        environment: the prefix of the environment the job runs in, whose dist-infos say where a
            distribution's compiled half was installed.
        needs: data paths declared at dispatch time, joining the ones the file declares.
        """
        sources = SourceTree(root)
        config = cls.__pytest_config(target, root) if target.test else ""
        home = cls.__placed(target.file, root=root, config=config)
        walker = Walker(root, home=home, distributions=distributions)
        reached = walker.reach(target.file)
        places = [home, *distributions]
        if target.test:
            for conftest, plugins in cls.__pytest_harness(target, root):
                place = cls.__placed(conftest, root=root, config=config)
                places.append(place)
                reached.extend(
                    Walker(root, home=place, distributions=distributions).reach(conftest)
                )
                for plugin in plugins:
                    reached.extend(walker.resolve(plugin))
        declared = target.declaration(root)
        files = set(sources.kept(target.node))
        built: set[str] = set()
        deferred: set[str] = set()
        decided: dict[str, tuple[bool, tuple[str, ...]]] = {}
        for module in reached:
            if module.root not in distributions:
                files.add(module.path)
                continue
            package = cls.__package(module)
            if package not in decided:
                decided[package] = cls.__compiled(package, root=root, environment=environment)
            outside, extensions = decided[package]
            if outside:
                deferred.add(PurePosixPath(package).name)
            else:
                files.update(sources.kept(package))
                built.update(extensions)
        files.update(built)
        for resource in declared.resources:
            files.update(cls.__pinned(resource, root, sources))
        files.add(Project().manifest)
        if config:
            files.add(config)
        wanted, pinned = split(dict.fromkeys([*declared.needs, *needs]))
        cls.__admissible(wanted, files)
        for pin in pinned:
            Pin.parse(pin)
        roster = tuple(dict.fromkeys(places))
        return cls(
            target=target,
            files=tuple(sorted(files)),
            roots=tuple(
                place
                for place in roster
                if place == "."
                or any(file == place or file.startswith(f"{place}/") for file in files)
            ),
            first_party=tuple(
                sorted({name for place in roster for name in _defined(root / place)})
            ),
            needs=wanted,
            pins=pinned,
            fetch=declared.fetch,
            built=tuple(sorted(built)),
            deferred=tuple(sorted(deferred)),
        )

    @staticmethod
    def __placed(file: str, *, root: Path, config: str) -> str:
        """`file`'s import root, workspace-relative, never above the pytest config's directory."""
        home = home_of(root / file, root=root)
        if config and home.is_relative_to((root / config).parent):
            home = (root / config).parent
        return home.relative_to(root).as_posix()

    @staticmethod
    def __pinned(resource: str, root: Path, sources: SourceTree) -> list[str]:
        """The files a declared resource pins: itself, or everything kept under it."""
        posix = PurePosixPath(resource)
        if posix.is_absolute() or ".." in posix.parts:
            raise MissionError(f"a resource must be a workspace-relative path, not {resource!r}")
        if (root / resource).is_dir():
            return sources.kept(resource)
        if (root / resource).is_file():
            return [posix.as_posix()]
        raise MissionError(f"the declared resource {resource!r} is not in the workspace")

    @staticmethod
    def __package(module: Module) -> str:
        """The top-level package directory `module` belongs to under its distribution root."""
        inside = PurePosixPath(module.path).relative_to(module.root)
        return (PurePosixPath(module.root) / inside.parts[0]).as_posix()

    @staticmethod
    def __compiled(package: str, *, root: Path, environment: Path) -> tuple[bool, tuple[str, ...]]:
        """Whether top-level `package` (workspace-relative) defers to the environment, and which
        of its extensions ship inside it."""
        tree = (root / package).resolve()
        inside: list[str] = []
        for file in recorded_extensions(PurePosixPath(package).name, prefix=environment):
            try:
                relative = file.resolve().relative_to(tree)
            except ValueError:
                return True, ()
            inside.append((PurePosixPath(package) / relative).as_posix())
        return False, tuple(inside)

    @staticmethod
    def __pytest_harness(target: Target, root: Path) -> Iterator[tuple[str, tuple[str, ...]]]:
        """The test file and every conftest above the node, with the plugins each names."""
        place = PurePosixPath(target.node)
        for candidate in [
            PurePosixPath(target.file),
            *(item / "conftest.py" for item in (place, *place.parents)),
        ]:
            if (root / candidate).is_file():
                yield candidate.as_posix(), _pytest_plugins(root / candidate)

    @staticmethod
    def __pytest_config(target: Target, root: Path) -> str:
        """The pytest configuration governing the node, nearest first as pytest adopts it, empty
        for none."""
        place = PurePosixPath(target.node)
        for item in (place, *place.parents):
            for name, carries in _PYTEST_CONFIGS:
                candidate = item / name
                if (root / candidate).is_file() and carries(root / candidate):
                    return candidate.as_posix()
        return ""

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


# The variable a conftest names its plugins by.
_PLUGINS = "pytest_plugins"


def _carries_pytest_options(file: Path) -> bool:
    """Whether a `pyproject.toml` carries a `[tool.pytest.ini_options]` table."""
    try:
        return "ini_options" in tomllib.loads(file.read_text(encoding="utf-8"))["tool"]["pytest"]
    except KeyError, TypeError, tomllib.TOMLDecodeError:
        return False


def _has_ini_section(file: Path, section: str) -> bool:
    """Whether an ini file carries `section`, read without importing anything."""
    parser = ConfigParser(interpolation=None)
    parser.read(file, encoding="utf-8")
    return parser.has_section(section)


# The config files pytest adopts, nearest first inside a directory, each judged by name or by
# a static read of the section it carries.
_PYTEST_CONFIGS: tuple[tuple[str, Callable[[Path], bool]], ...] = (
    ("pytest.ini", lambda _: True),
    ("pyproject.toml", _carries_pytest_options),
    ("tox.ini", partial(_has_ini_section, section="pytest")),
    ("setup.cfg", partial(_has_ini_section, section="tool:pytest")),
)


def _pytest_plugins(file: Path) -> tuple[str, ...]:
    """The plugin modules `file` names in a literal `pytest_plugins` (a string or a sequence).

    Anything needing execution to be known is refused rather than guessed at.
    """
    for node in parsed(file).body:
        if (value := _plugins_value(node)) is None:
            continue
        try:
            literal = ast.literal_eval(value)
        except ValueError as opaque:
            raise MissionError(
                f"`{_PLUGINS}` in {file} must be a literal module name or list of them, since "
                f"the file is read without being imported; {opaque}"
            ) from None
        names = (literal,) if isinstance(literal, str) else tuple(literal)
        if not all(isinstance(name, str) for name in names):
            raise MissionError(
                f"`{_PLUGINS}` in {file} must name modules as strings, not {names!r}"
            )
        return names
    return ()


def _plugins_value(node: ast.stmt) -> ast.expr | None:
    """The value `node` assigns to `_PLUGINS`, None when it is not that assignment."""
    match node:
        case ast.Assign(targets=targets, value=value) if any(
            isinstance(target, ast.Name) and target.id == _PLUGINS for target in targets
        ):
            return value
        case ast.AnnAssign(target=ast.Name(id=name), value=value) if name == _PLUGINS:
            return value
    return None


def _absolute(package: str, *, level: int, name: str) -> str | None:
    """`name` (empty for `from . import x`) imported from module `package` with `level` leading
    dots, None past the top package."""
    if not level:
        return name
    parts = package.split(".")[:-1]
    if level - 1 >= len(parts):
        return None
    base = parts[: len(parts) - (level - 1)]
    return ".".join([*base, name] if name else base)


def _dynamic(call: ast.Call) -> str:
    """The absolute literal `import_module("a.b")` or `__import__("a.b")` imports, else empty."""
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
