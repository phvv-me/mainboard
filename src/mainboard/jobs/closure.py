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
from collections.abc import Set as AbstractSet
from configparser import ConfigParser
from functools import partial
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, NamedTuple

from patos import FrozenModel

from ..core.errors import MissionError
from ..core.project import Project
from ..dispatch.provenance import SourceTree
from ..engines.compile.backend.pixi.repair import recorded_extensions
from ..state.evidence import EvidenceTree
from .targets.pins import Pin, split
from .targets.target import Target, home_of, parsed
from .walker import Module, Walker

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence


class _Shipped(NamedTuple):
    """What a job's reached modules ship: files, the extensions among them, deferred names."""

    files: set[str]
    built: set[str]
    deferred: set[str]


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
    def __needed(
        cls, declared: Sequence[str], *, files: AbstractSet[str], root: Path
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """The data paths and Hub pins `declared` needs, each checked, the paths in the tree.

        A need is data the mirror carries to the host, so one only the lake still keeps comes
        back into the tree before the sync.
        """
        spelled, pinned = split(dict.fromkeys(declared))
        wanted = cls.__outermost(spelled)
        cls.__admissible(wanted, files)
        EvidenceTree(root).restore(root / need for need in wanted if not (root / need).exists())
        for pin in pinned:
            Pin.parse(pin)
        return wanted, pinned

    @classmethod
    def __reached(
        cls, target: Target, *, root: Path, config: str, distributions: Sequence[str]
    ) -> tuple[list[Module], list[str]]:
        """Every first-party module the target imports, and the import roots they came from.

        A pytest target also reaches each conftest above it with its imports and named plugins.
        """
        home = cls.__placed(target.file, root=root, config=config)
        walker = Walker(root, home=home, distributions=distributions)
        reached, places = walker.reach(target.file), [home, *distributions]
        if not target.test:
            return reached, places
        for conftest, plugins in cls.__pytest_harness(target, root):
            place = cls.__placed(conftest, root=root, config=config)
            places.append(place)
            reached += Walker(root, home=place, distributions=distributions).reach(conftest)
            reached += [module for plugin in plugins for module in walker.resolve(plugin)]
        return reached, places

    @classmethod
    def __shipped(
        cls,
        reached: Sequence[Module],
        *,
        sources: SourceTree,
        distributions: Sequence[str],
        environment: Path,
    ) -> _Shipped:
        """What the reached modules ship.

        A node's module ships itself, a distribution's module its whole top-level package with
        the extensions built inside it, unless an extension lives outside it, which defers the
        package to the environment.
        """
        shipped = _Shipped(
            files={module.path for module in reached if module.root not in distributions},
            built=set(),
            deferred=set(),
        )
        packages = dict.fromkeys(cls.__package(m) for m in reached if m.root in distributions)
        for package in packages:
            outside, extensions = cls.__compiled(
                package, root=sources.root, environment=environment
            )
            if outside:
                shipped.deferred.add(PurePosixPath(package).name)
            else:
                shipped.files.update([*sources.kept(package), *extensions])
                shipped.built.update(extensions)
        return shipped

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
        reached, places = cls.__reached(
            target, root=root, config=config, distributions=distributions
        )
        shipped = cls.__shipped(
            reached, sources=sources, distributions=distributions, environment=environment
        )
        declared = target.declaration(root)
        files = {
            *sources.kept(target.node),
            *shipped.files,
            *(
                found
                for resource in declared.resources
                for found in cls.__pinned(resource, root, sources)
            ),
            Project().manifest(root).name,
            *([config] if config else []),
        }
        wanted, pinned = cls.__needed([*declared.needs, *needs], files=files, root=root)
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
                sorted({name for place in roster for name in Closure._defined(root / place)})
            ),
            needs=wanted,
            pins=pinned,
            fetch=declared.fetch,
            built=tuple(sorted(shipped.built)),
            deferred=tuple(sorted(shipped.deferred)),
        )

    @staticmethod
    def __admissible(needs: Sequence[str], files: AbstractSet[str]) -> None:
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

    @staticmethod
    def __compiled(package: str, *, root: Path, environment: Path) -> tuple[bool, tuple[str, ...]]:
        """Whether top-level `package` defers to the environment, and which extensions ship.

        `package` is workspace-relative, and the extensions are those that ship inside it.
        """
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
    def __outermost(needs: Sequence[str]) -> tuple[str, ...]:
        """Each need in its normal spelling, less those under another need.

        The host links each need back to the mirror, so one beneath a linked need would be placed
        through that link, over the mirror's own file.
        """
        spelled = tuple(dict.fromkeys(PurePosixPath(need).as_posix() for need in needs))
        return tuple(
            need
            for need in spelled
            if not any(parent.as_posix() in spelled for parent in PurePosixPath(need).parents)
        )

    @staticmethod
    def __package(module: Module) -> str:
        """The top-level package directory `module` belongs to under its distribution root."""
        inside = PurePosixPath(module.path).relative_to(module.root)
        return (PurePosixPath(module.root) / inside.parts[0]).as_posix()

    @staticmethod
    def __pinned(resource: str, root: Path, sources: SourceTree) -> list[str]:
        """The files a declared resource pins: itself, or everything kept under it.

        They are brought back from the lake first when the tree no longer holds them.
        """
        posix = PurePosixPath(resource)
        if posix.is_absolute() or ".." in posix.parts:
            raise MissionError(f"a resource must be a workspace-relative path, not {resource!r}")
        if not (root / resource).exists():
            EvidenceTree(root).restore([root / resource])
        if (root / resource).is_dir():
            return sources.kept(resource)
        if (root / resource).is_file():
            return [posix.as_posix()]
        raise MissionError(f"the declared resource {resource!r} is not in the workspace")

    @staticmethod
    def __placed(file: str, *, root: Path, config: str) -> str:
        """`file`'s import root, workspace-relative, never above the pytest config's directory."""
        home = home_of(root / file, root=root)
        if config and home.is_relative_to((root / config).parent):
            home = (root / config).parent
        return home.relative_to(root).as_posix()

    @staticmethod
    def __pytest_config(target: Target, root: Path) -> str:
        """The pytest configuration governing the node, empty for none.

        The nearest is taken first, as pytest adopts it.
        """
        place = PurePosixPath(target.node)
        for item in (place, *place.parents):
            for name, carries in _PYTEST_CONFIGS:
                candidate = item / name
                if (root / candidate).is_file() and carries(root / candidate):
                    return candidate.as_posix()
        return ""

    @staticmethod
    def __pytest_harness(target: Target, root: Path) -> Iterator[tuple[str, tuple[str, ...]]]:
        """The test file and every conftest above the node, with the plugins each names."""
        place = PurePosixPath(target.node)
        for candidate in [
            PurePosixPath(target.file),
            *(item / "conftest.py" for item in (place, *place.parents)),
        ]:
            if (root / candidate).is_file():
                yield candidate.as_posix(), Closure._pytest_plugins(root / candidate)

    @staticmethod
    def _defined(root: Path) -> set[str]:
        """The top-level names importable from `root`.

        Those are its packages, namespace portions included, and its bare modules. A portion is a
        directory without an initializer holding a module or a package, so code in one is guarded
        as a regular package's is.
        """
        try:
            entries = list(root.iterdir())
        except OSError:
            return set()
        return {
            entry.name
            for entry in entries
            if entry.name.isidentifier()
            and entry.is_dir()
            and (
                (entry / "__init__.py").is_file()
                or any(
                    child.suffix == ".py" or (child / "__init__.py").is_file()
                    for child in entry.iterdir()
                )
            )
        } | {entry.stem for entry in entries if entry.suffix == ".py" and entry.is_file()}

    @staticmethod
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

    @staticmethod
    def _pytest_plugins(file: Path) -> tuple[str, ...]:
        """The plugin modules `file` names in a literal `pytest_plugins` (a string or a sequence).

        Anything needing execution to be known is refused rather than guessed at.
        """
        for node in parsed(file).body:
            if (value := Closure._plugins_value(node)) is None:
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
