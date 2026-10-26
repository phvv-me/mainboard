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
# them for when it stands alone; composed, they are not read, and `doctor --members` names them.
#
# NAMES. A member is named after its directory. Its tasks and papers are reachable as
# `<member>:<name>`, and bare as well when neither the root nor another member takes the name;
# its lint tools join as `<member>:<tool>` over its own files unless the root declares that tool.
# Every local path in a member's manifest is relative to the member and is rebased on the way in.

from functools import cached_property
from pathlib import PurePosixPath
from typing import TYPE_CHECKING

from packaging.utils import canonicalize_name

from ...core.errors import MissionError
from ...core.membership import Membership
from ...core.project import Project
from ..parsing import rendered, validated
from ..schema.compute.toolchain import Toolchain
from ..schema.root import Manifest
from ..schema.scope import PlatformScope, Scope
from ..schema.spec import Spec
from .models import Member, Package
from .rebasing import anchored, moved, namespaced, rebased, sources_rebased

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping
    from pathlib import Path

    from ..render.interpolate import Json
    from ..schema.environment import Env
    from ..schema.project.lint import Lint

# The tables only a workspace root reads. `[workspace]` is always a member's standalone header
# and is never reported; `[vars]` has already rendered into the member's own strings.
_ROOT_ONLY = (
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

_OVERRIDES = "dependency-overrides"
_PYTHON = "python"


class Composition:
    """A workspace manifest and the members its `[workspace] members` declares, folded into one."""

    @cached_property
    def _sources(self) -> dict[str, dict[str, Json]]:
        """Every member's requirement on itself, its directory installed editable, by name."""
        return {
            self._spelled(package.name): {"path": member.path, "editable": True}
            for member, package in self._packaged()
        }

    @cached_property
    def _spellings(self) -> dict[str, str]:
        """The root's own spelling of every Python distribution it names, by normalized name."""
        root = [*_python(self.manifest).all_deps(), *_python(self.manifest.dev).all_deps()]
        return {canonicalize_name(spelling): spelling for spelling in root}

    @cached_property
    def members(self) -> list[Member]:
        """Every member, in path order, refusing two that would share a namespace."""
        paths = Membership(
            self.root, self.manifest.workspace.members, manifests=Project().manifests
        )
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

    @staticmethod
    def _ignored(manifest: Manifest) -> tuple[str, ...]:
        """The root-only settings `manifest` declares, as a reader of its file would name them."""
        lint = manifest.lint.model_fields_set
        return (
            *(f"[{table}]" for table in _ROOT_ONLY if table in manifest.model_fields_set),
            *(
                f"[{name}] {option}"
                for name, chain in manifest.toolchains().items()
                for option in chain.model_extra or {}
            ),
            *(f"[lint] {field.replace('_', '-')}" for field in _ROOT_LINT if field in lint),
        )

    def __init__(self, root: Path, manifest: Manifest) -> None:
        self.root = root
        self.manifest = manifest

    def _alone(self, env: Env, pins: Mapping[str, Json]) -> Env:
        """`env` with its member requirements sourced and every member pinned to its directory."""
        if not pins:
            return env
        sourced = self._sourced(env)
        chain = _python(sourced).model_dump(mode="python", round_trip=True)
        # The validated toolchain dump drops the env's own pypi-options (dependency-overrides,
        # index urls) when its `[python]` table carries no `deps`/`dev`, so carry them from the raw
        # table: a lean env's own override must survive composition, not only the member pins.
        own = (sourced.model_extra or {}).get(_PYTHON, {})
        for key, value in (own if isinstance(own, dict) else {}).items():
            if key not in ("deps", "dev"):
                chain.setdefault(key, value)
        # The extras the top level unions are what `default` asks; a lean environment asks only
        # through its own requirements, so it pins each member's directory alone.
        bare = {
            name: {"path": pin["path"]} if isinstance(pin, dict) and "path" in pin else pin
            for name, pin in pins.items()
        }
        chain[_OVERRIDES] = bare | chain.get(_OVERRIDES, {})
        return sourced.model_copy(update={_PYTHON: chain})

    def _declared(self) -> list[tuple[Member, Manifest]]:
        """Every member with a manifest of its own, beside that manifest."""
        return [(member, member.manifest) for member in self.members if member.manifest]

    def _environments(self, pins: Mapping[str, Json]) -> dict[str, Env]:
        """The root's named environments and every member's; one declared by several joins them.

        A member's environment of a name the root (or an earlier member) declares contributes
        to it, as its `[deps]` contribute to `default`: llm-head's `serving` adds its serving
        extra and task to the root's, the root's settings and bounds winning (`Env.merged`).

        An environment the root declares that starts from nothing (`no-default`) leaves the
        default feature out, and the member pins with it, so each gets them again: a member it
        names, or one its closure reaches (cutoken needs mainboard), comes from the workspace,
        never an index.

        pins: the member overrides the top level carries.
        """
        environments = dict(self.manifest.envs)
        for member, manifest in self._declared():
            for name, env in manifest.envs.items():
                relocated = moved(member, env.tasks, siblings=manifest.tasks | env.tasks)
                tasks = namespaced({member.name: relocated}, root={})
                joined = env.model_copy(update={"tasks": tasks})
                environments[name] = (
                    environments[name].merged(joined) if name in environments else joined
                )
        return {
            name: self._alone(env, pins) if env.no_default and name in self.manifest.envs else env
            for name, env in environments.items()
        }

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

    def _lint(self) -> Lint:
        """The root's lint table, owning each member and taking its exclusions and new tools."""
        lint = self.manifest.lint
        tools = {
            f"{member.name}:{name}": tool.model_copy(
                update={
                    "files": anchored(tool.files, member.path),
                    "exclude": anchored(tool.exclude, member.path),
                }
            )
            for member, manifest in self._declared()
            for name, tool in manifest.lint.tools.items()
            if name not in lint.tools
        }
        excluded = [
            pattern
            for member, manifest in self._declared()
            for pattern in anchored(manifest.lint.exclude, member.path)
        ]
        return lint.model_copy(
            update={
                "owners": (*lint.owners, *(member.path for member in self.members)),
                "exclude": (*lint.exclude, *excluded),
                "tools": lint.tools | tools,
            }
        )

    def _member(self, path: str) -> Member:
        """The member at workspace-relative `path`, its own manifest read if it has one."""
        directory = self.root / path
        name = PurePosixPath(path).name
        package = Package.read(directory / Package.PYPROJECT)
        source = Project().manifest(directory)
        try:
            tree = rendered(source)
        except FileNotFoundError:
            return Member(name=name, path=path, package=package)
        manifest = validated(source, sources_rebased(tree, path))
        return Member(
            name=name,
            path=path,
            package=package,
            manifest=manifest,
            ignored=Composition._ignored(manifest),
        )

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

    def _packaged(self) -> list[tuple[Member, Package]]:
        """Every member with an installable Python project, beside that project."""
        return [(member, member.package) for member in self.members if member.package]

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

    def _spelled(self, name: str) -> str:
        """How the root spells distribution `name`, normalized when the root does not name it.

        One distribution spelled two ways (`llm_head`, `llm-head`) would reach the solver twice.
        """
        normalized = canonicalize_name(name)
        return self._spellings.get(normalized, normalized)

    def composed(self) -> Manifest:
        """The root with every member folded in, the root layered over them wherever both speak."""
        if not self.members:
            return self.manifest
        top, dev, overlays = self._requirements()
        # Only what the root declared, so a profile still inherits `[hosts.defaults]` for every
        # field it left unset.
        body = self.manifest.model_dump(mode="python", round_trip=True, exclude_unset=True)
        body.update(top.model_dump(mode="python", round_trip=True))
        pins = (_python(top).model_extra or {}).get(_OVERRIDES, {})
        body |= {
            "workspace": self.manifest.workspace,
            "dev": dev,
            "on": overlays,
            "env": {
                name: value
                for _, manifest in self._declared()
                for name, value in manifest.env.items()
            }
            | self.manifest.env,
            "envs": self._environments(pins),
            "tasks": namespaced(
                {
                    member.name: moved(member, manifest.tasks, siblings=manifest.tasks)
                    for member, manifest in self._declared()
                },
                root=self.manifest.tasks,
            ),
            "papers": namespaced(
                {
                    member.name: {
                        name: paper.model_copy(
                            update={"dir": rebased(member.path, path=paper.dir)}
                        )
                        for name, paper in manifest.papers.items()
                    }
                    for member, manifest in self._declared()
                },
                root=self.manifest.papers,
            ),
            "lint": self._lint(),
        }
        return Manifest.model_validate(body)


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
