# HOW A WORKSPACE COMPOSES ITS MEMBERS. A member is a project that is complete on its own: a
# `pyproject.toml` anybody can `pip install`, whose dependencies name house packages the normal
# way (a version or a git URL), and optionally its own manifest for what only it needs (tasks,
# papers, lint, system packages). Cloned alone, that manifest is the workspace. Inside the
# monorepo, the root's `[workspace] members` globs name it and `load` folds it into the root.
#
# WHAT JOINS. Requirements (conda and every ecosystem's `deps` and `dev`, platform overlays,
# `[dev]`), `[env]`, named environments, tasks, papers and lint. The root is layered on top
# wherever both speak, with one exception: a member with a `pyproject.toml` is always its own
# source. Wherever the workspace requires it, the root included, the requirement is its
# directory installed editable, keeping its extras (a member nothing requires joins
# `[python.deps]`), and a dependency override pins it there, so a requirement on it anywhere in
# the closure, a version, an exact pin or a git URL, resolves to the local source, the way
# cargo's `[patch]` does.
#
# WHAT STAYS OUT. How to solve and where to run belong to the root alone: `[workspace]`,
# `[system]`, each ecosystem's solve settings (`[python] index-strategy`, overrides), hosts,
# admission, containers, CI, git, gates, templates, tracking and plot styles. A member keeps
# them for when it stands alone; composed, they are not read, and `center members` names them.
#
# NAMES. A member is named after its directory. Its tasks and papers are reachable as
# `<member>:<name>`, and bare as well when neither the root nor another member takes the name;
# its lint tools join as `<member>:<tool>` over its own files unless the root declares that tool.
# Every local path in a member's manifest is relative to the member and is rebased on the way in.

import posixpath
import tomllib
from collections import Counter
from functools import cached_property
from pathlib import PurePosixPath, PureWindowsPath
from typing import TYPE_CHECKING

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from patos import FrozenModel

from ..core.errors import MissionError
from ..core.membership import Membership
from ..core.project import Project
from .parsing import rendered, validated
from .schema.root import Manifest
from .schema.scope import PlatformScope, Scope
from .schema.spec import Spec
from .schema.toolchain import Toolchain

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping
    from pathlib import Path

    from .render.interpolate import Json
    from .schema.environment import Env, Task
    from .schema.lint import Lint

PYPROJECT = "pyproject.toml"

# The tables only a workspace root reads. `[workspace]` is always a member's standalone header
# and is never reported; `[vars]` has already rendered into the member's own strings.
ROOT_ONLY = (
    "admission",
    "ci",
    "containers",
    "figures",
    "gates",
    "git",
    "hosts",
    "plots",
    "system",
    "templates",
    "tracking",
)

# The `[lint]` settings a member's files are judged by only through the root's.
_ROOT_LINT = ("owners", "markers", "max_kb")

# The tables whose entries are dependency specs, where a local `path` source can appear.
_SPEC_TABLES = frozenset({"deps", "dev", "dependency-overrides"})

_OVERRIDES = "dependency-overrides"
_PYTHON = "python"


class Package(FrozenModel):
    """A member's installable Python project, as its `pyproject.toml` declares it.

    name: the distribution, normalized.
    python: its `requires-python`, empty when unstated.
    requires: every distribution any of its dependency lists or groups names, normalized, with
        every extra asked of it there.
    """

    name: str
    python: str = ""
    requires: dict[str, frozenset[str]] = {}

    @classmethod
    def read(cls, path: Path) -> Package | None:
        """The project `path` declares, None when the file or its `[project]` table is absent."""
        try:
            tree = tomllib.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except tomllib.TOMLDecodeError as error:
            raise MissionError(f"{path} is not valid TOML: {error}") from None
        project = tree.get("project", {})
        if "name" not in project:
            return None
        lists = [
            project.get("dependencies", []),
            *project.get("optional-dependencies", {}).values(),
            *tree.get("dependency-groups", {}).values(),
        ]
        requires: dict[str, frozenset[str]] = {}
        for requirement in (
            Requirement(line) for listed in lists for line in listed if isinstance(line, str)
        ):
            name = canonicalize_name(requirement.name)
            requires[name] = requires.get(name, frozenset()) | requirement.extras
        return cls(
            name=canonicalize_name(project["name"]),
            python=project.get("requires-python", ""),
            requires=requires,
        )


class Member(FrozenModel):
    """One project a workspace composes.

    name: its directory's name, the namespace of its tasks, papers and lint tools.
    path: workspace-relative.
    package: its installable Python project, None without one.
    manifest: its own manifest with every local path rebased onto the workspace, None without.
    ignored: the root-only settings its manifest declares, read only when it stands alone.
    """

    name: str
    path: str
    package: Package | None = None
    manifest: Manifest | None = None
    ignored: tuple[str, ...] = ()


class Composition:
    """A workspace manifest and the members its `[workspace] members` declares, folded into one."""

    def __init__(self, root: Path, manifest: Manifest) -> None:
        self.root = root
        self.manifest = manifest

    @cached_property
    def members(self) -> list[Member]:
        """Every member, in path order, refusing two that would share a namespace."""
        paths = Membership(self.root, self.manifest.workspace.members, Project().manifest)
        members = [self._member(path) for path in paths.directories()]
        seen: dict[str, str] = {}
        for member in members:
            if member.name in seen:
                raise MissionError(
                    f"members {seen[member.name]} and {member.path} are both named "
                    f"{member.name!r}, which namespaces their tasks; leave one out with `!`"
                )
            seen[member.name] = member.path
        return members

    def composed(self) -> Manifest:
        """The root with every member folded in, the root layered over them wherever both speak."""
        if not self.members:
            return self.manifest
        top, dev, overlays = self._requirements()
        # Only what the root declared, so a profile still inherits `[hosts.defaults]` for every
        # field it left unset.
        body = self.manifest.model_dump(mode="python", round_trip=True, exclude_unset=True)
        body.update(top.model_dump(mode="python", round_trip=True))
        body["workspace"] = self.manifest.workspace
        body["dev"] = dev
        body["on"] = overlays
        body["env"] = {
            name: value for _, manifest in self._declared() for name, value in manifest.env.items()
        } | self.manifest.env
        body["envs"] = self._environments()
        body["tasks"] = _namespaced(
            {
                member.name: _moved(member, manifest.tasks, siblings=manifest.tasks)
                for member, manifest in self._declared()
            },
            root=self.manifest.tasks,
        )
        body["papers"] = _namespaced(
            {
                member.name: {
                    name: paper.model_copy(update={"dir": _rebased(member.path, paper.dir)})
                    for name, paper in manifest.papers.items()
                }
                for member, manifest in self._declared()
            },
            root=self.manifest.papers,
        )
        body["lint"] = self._lint()
        return Manifest.model_validate(body)

    def _member(self, path: str) -> Member:
        """The member at workspace-relative `path`, its own manifest read if it has one."""
        directory = self.root / path
        name = PurePosixPath(path).name
        package = Package.read(directory / PYPROJECT)
        source = directory / Project().manifest
        try:
            tree = rendered(source)
        except FileNotFoundError:
            return Member(name=name, path=path, package=package)
        manifest = validated(source, _sources_rebased(tree, path))
        return Member(
            name=name, path=path, package=package, manifest=manifest, ignored=_ignored(manifest)
        )

    def _declared(self) -> list[tuple[Member, Manifest]]:
        """Every member with a manifest of its own, beside that manifest."""
        return [(member, member.manifest) for member in self.members if member.manifest]

    def _packaged(self) -> list[tuple[Member, Package]]:
        """Every member with an installable Python project, beside that project."""
        return [(member, member.package) for member in self.members if member.package]

    def _requirements(self) -> tuple[Scope, Scope, dict[str, PlatformScope]]:
        """The composed top-level, `[dev]` and platform scopes, every member from its source.

        Wherever any scope requires a member, the root included, that requirement becomes the
        member's own directory, editable, keeping its extras; a member nothing requires joins
        the top level. Each source is then pinned by an override on the top level, so a
        requirement on a member anywhere in the closure resolves to its directory too.
        """
        manifests = [manifest for _, manifest in self._declared()]
        top = _view(self.manifest).merged(self._folded(map(_view, manifests), Scope()))
        dev = self.manifest.dev.merged(self._folded((m.dev for m in manifests), Scope()))
        scopes = [self._sourced(top), self._sourced(dev)]
        overlays = {platform: self._sourced(scope) for platform, scope in self._overlays().items()}
        required = {
            name for scope in (*scopes, *overlays.values()) for name in _python(scope).all_deps()
        }
        unrequired = {
            name: source for name, source in self._sources.items() if name not in required
        }
        if unrequired:
            scopes[0] = Scope.model_validate({_PYTHON: {"deps": unrequired}}).merged(scopes[0])
        top, dev = scopes
        return self._pinned(top, [top, dev, *overlays.values()]), dev, overlays

    @cached_property
    def _sources(self) -> dict[str, dict[str, Json]]:
        """Every member's requirement on itself, its directory installed editable, by name."""
        return {
            self._spelled(package.name): {"path": member.path, "editable": True}
            for member, package in self._packaged()
        }

    def _sourced[S: Scope](self, scope: S) -> S:
        """`scope` with each Python requirement on a member taken from the member's directory."""
        python = scope.toolchains().get(_PYTHON)
        if python is None:
            return scope
        tables = {
            "deps": self._sourced_specs(python.deps),
            "dev": self._sourced_specs(python.dev),
        }
        chain = python.model_dump(mode="python", round_trip=True) | tables
        return scope.model_copy(update={_PYTHON: chain})

    def _sourced_specs(self, specs: Mapping[str, Spec]) -> dict[str, Json]:
        """`specs`, each one naming a member layered under that member's source."""
        return {
            name: (
                Spec.model_validate(self._sources[name]).merged(spec)
                if name in self._sources
                else spec
            ).model_dump(mode="python", round_trip=True)
            for name, spec in specs.items()
        }

    def _pinned(self, top: Scope, scopes: Iterable[Scope]) -> Scope:
        """`top` with every member source pinned by an override, the root's own ones winning.

        An override replaces every requirement it matches, extras included, so it carries every
        extra any of `scopes` or any other member's project asks of that member; a project
        naming itself (a test group installing its own extras) asks nothing of the workspace.
        """
        requirements = [python.all_deps() for python in map(_python, scopes)]
        generated: dict[str, Json] = {}
        for member, package in self._packaged():
            name = self._spelled(package.name)
            declared = {
                extra
                for specs in requirements
                if name in specs
                for extra in (specs[name].model_extra or {}).get("extras", [])
            }
            asked = (
                other.requires.get(package.name, ())
                for sibling, other in self._packaged()
                if sibling is not member
            )
            extras = sorted(declared.union(*asked))
            generated[name] = {"path": member.path, **({"extras": extras} if extras else {})}
        if not generated:
            return top
        python = _python(top)
        chain = python.model_dump(mode="python", round_trip=True)
        chain[_OVERRIDES] = generated | (python.model_extra or {}).get(_OVERRIDES, {})
        return Scope.model_validate(
            {**top.model_dump(mode="python", round_trip=True), _PYTHON: chain}
        )

    def _spelled(self, name: str) -> str:
        """How the root spells distribution `name`, normalized when the root does not name it.

        One distribution spelled two ways (`llm_head`, `llm-head`) would reach the solver twice.
        """
        normalized = canonicalize_name(name)
        return self._spellings.get(normalized, normalized)

    @cached_property
    def _spellings(self) -> dict[str, str]:
        """The root's own spelling of every Python distribution it names, by normalized name."""
        root = [*_python(self.manifest).all_deps(), *_python(self.manifest.dev).all_deps()]
        return {canonicalize_name(spelling): spelling for spelling in root}

    def _folded[S: Scope](self, scopes: Iterable[S], empty: S) -> S:
        """What every member requires in `scopes`, a later member layered over an earlier one.

        Each ecosystem's solve settings stay out, since they are the root's alone.

        empty: the scope folding starts from, which fixes the kind every step returns.
        """
        layered = empty
        for scope in scopes:
            chains = {
                name: chain.model_dump(mode="python", round_trip=True, include={"deps", "dev"})
                for name, chain in scope.toolchains().items()
            }
            if python := chains.get(_PYTHON):
                chains[_PYTHON] = {
                    table: {self._spelled(name): spec for name, spec in specs.items()}
                    for table, specs in python.items()
                }
            own = scope.model_dump(
                mode="python", round_trip=True, exclude=set(scope.model_extra or {})
            )
            layered = scope.model_validate({**own, **chains}).merged(layered)
        return layered

    def _overlays(self) -> dict[str, PlatformScope]:
        """Each platform overlay, the root's over every member's requirements for that platform."""
        manifests = [manifest for _, manifest in self._declared()]
        platforms = sorted({platform for m in (self.manifest, *manifests) for platform in m.on})
        return {
            platform: self.manifest.on.get(platform, PlatformScope()).merged(
                self._folded(
                    (m.on[platform] for m in manifests if platform in m.on), PlatformScope()
                )
            )
            for platform in platforms
        }

    def _environments(self) -> dict[str, Env]:
        """The root's named environments and every member's, a name declared twice refused."""
        environments = dict(self.manifest.envs)
        owners = dict.fromkeys(environments, "the root")
        for member, manifest in self._declared():
            for name, env in manifest.envs.items():
                if name in owners:
                    raise MissionError(
                        f"{member.path} declares environment {name!r}, which {owners[name]} "
                        "declares too; rename one of them"
                    )
                owners[name] = member.path
                moved = _moved(member, env.tasks, siblings=manifest.tasks | env.tasks)
                tasks = _namespaced({member.name: moved}, root={})
                environments[name] = env.model_copy(update={"tasks": tasks})
        return environments

    def _lint(self) -> Lint:
        """The root's lint table, owning each member and taking its exclusions and new tools."""
        lint = self.manifest.lint
        tools = {
            f"{member.name}:{name}": tool.model_copy(
                update={
                    "files": _anchored(tool.files, member.path),
                    "exclude": _anchored(tool.exclude, member.path),
                }
            )
            for member, manifest in self._declared()
            for name, tool in manifest.lint.tools.items()
            if name not in lint.tools
        }
        excluded = [
            pattern
            for member, manifest in self._declared()
            for pattern in _anchored(manifest.lint.exclude, member.path)
        ]
        return lint.model_copy(
            update={
                "owners": (*lint.owners, *(member.path for member in self.members)),
                "exclude": (*lint.exclude, *excluded),
                "tools": lint.tools | tools,
            }
        )


def _python(scope: Scope) -> Toolchain:
    """The Python ecosystem table of `scope`, empty when it declares none."""
    return scope.toolchains().get(_PYTHON, Toolchain())


def _view(manifest: Manifest) -> Scope:
    """The top-level requirements of `manifest` as a scope that layers like any other."""
    chains = {
        name: chain.model_dump(mode="python", round_trip=True, exclude_defaults=True)
        for name, chain in manifest.toolchains().items()
    }
    return Scope.model_validate({"deps": manifest.deps, **chains})


def _namespaced[T](
    declared: Mapping[str, Mapping[str, T]], *, root: Mapping[str, T]
) -> dict[str, T]:
    """Every member entry as `<member>:<name>`, and bare where nobody else takes the name.

    declared: each member's entries by member name.
    root: the root's own entries, which keep their names and win over any bare member one.
    """
    counts = Counter(name for entries in declared.values() for name in entries)
    named: dict[str, T] = {}
    for member, entries in declared.items():
        for name, entry in entries.items():
            named[f"{member}:{name}"] = entry
            if counts[name] == 1:
                named[name] = entry
    return named | dict(root)


def _moved(
    member: Member, tasks: Mapping[str, Task], *, siblings: Mapping[str, Task]
) -> dict[str, Task]:
    """`member`'s tasks running from its directory, each dependency on a sibling renamed.

    siblings: every task of the member a dependency can name; any other stays the root's.
    """
    moved: dict[str, Task] = {}
    for name, spec in tasks.items():
        task: dict[str, str | list[str] | dict[str, str]] = (
            {"run": spec} if isinstance(spec, str) else dict(spec)
        )
        if "run" in task:
            task["dir"] = _rebased(member.path, str(task.get("dir", "")))
        if "depends" in task:
            task["depends"] = [
                f"{member.name}:{need}" if need in siblings else need for need in task["depends"]
            ]
        moved[name] = task
    return moved


def _rebased(base: str, path: str) -> str:
    """`path`, relative to the member at `base`, as the workspace spells it; absolute ones stay."""
    if PurePosixPath(path).is_absolute() or PureWindowsPath(path).is_absolute():
        return path
    return posixpath.normpath(posixpath.join(base, path))


def _sources_rebased(tree: dict[str, Json], base: str) -> dict[str, Json]:
    """`tree`, a member's manifest or a table in it, with every local dependency source rebased."""
    return {
        key: _specs_rebased(item, base)
        if key in _SPEC_TABLES
        else _sources_rebased(item, base)
        if isinstance(item, dict)
        else item
        for key, item in tree.items()
    }


def _specs_rebased(table: Json, base: str) -> Json:
    """A dependency table, or a scope nested in one (`[dev]`), with its `path` sources rebased."""
    if not isinstance(table, dict):
        return table
    return {
        name: {**spec, "path": _rebased(base, path)}
        if isinstance(spec, dict) and isinstance(path := spec.get("path"), str)
        else _specs_rebased(spec, base)
        for name, spec in table.items()
    }


def _anchored(patterns: Iterable[str], base: str) -> tuple[str, ...]:
    """Gitignore patterns written inside the member at `base`, as the workspace root reads them.

    A slash before the end already anchors a pattern to its own directory; one without matches
    at any depth, which beneath the member is `**/`.
    """
    anchored: list[str] = []
    for pattern in patterns:
        negated, bare = ("!", pattern[1:]) if pattern.startswith("!") else ("", pattern)
        rooted = "/" in bare.rstrip("/")
        anchored.append(f"{negated}/{base}/{bare.lstrip('/') if rooted else f'**/{bare}'}")
    return tuple(anchored)


def _ignored(manifest: Manifest) -> tuple[str, ...]:
    """The root-only settings `manifest` declares, as a reader of its file would name them."""
    lint = manifest.lint.model_fields_set
    return (
        *(f"[{table}]" for table in ROOT_ONLY if table in manifest.model_fields_set),
        *(
            f"[{name}] {option}"
            for name, chain in manifest.toolchains().items()
            for option in chain.model_extra or {}
        ),
        *(f"[lint] {field.replace('_', '-')}" for field in _ROOT_LINT if field in lint),
    )
