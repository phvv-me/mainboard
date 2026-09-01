from pathlib import PurePosixPath, PureWindowsPath
from typing import TYPE_CHECKING, Self

import tomlkit
import tomlkit.items
from patos import FrozenModel
from pydantic import Field

from .platforms import PlatformMatrix

# `Toml` backs pydantic fields below, so it must resolve at class-creation time. See the
# matching comment in platforms.py for why ruff's flake8-type-checking cannot tell.
from .toml import Toml

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from ...manifest import Env, Manifest, Scope, Spec, Toolchain
    from ...manifest.schema.environment import Task

# pixi tables whose values are dependency specs. A ``path`` *source* lives inside one of these
# specs, never at the dep-name level.
_DEP_TABLES = ("dependencies", "pypi-dependencies", "dependency-overrides")

_PLATFORMS = "platforms"
_PYPI_OPTIONS = "pypi-options"

# pixi's own `[pypi-options]` fields, the uv settings that shape the Python solve. Anything
# else declared beside `[python.deps]` configures something other than pixi (chefe's `indexes`
# aliases, for one) and is not pixi's to read. `dependency-overrides` comes last because it is
# the only sub-table here, and TOML reads every key after a sub-table header as belonging to
# that sub-table.
_PYPI_OPTION_KEYS = (
    "index-url",
    "extra-index-urls",
    "find-links",
    "index-strategy",
    "no-build-isolation",
    "no-build",
    "no-binary",
    "prerelease-mode",
    "dependency-overrides",
)

# The generated dotenv loader, sourced first by pixi activation when `workspace.dotenv` is on.
# Activation scripts run from the manifest dir (`.mainboard/`), so `../.env` is the workspace
# root.
_DOTENV_SH = "dotenv.sh"
_DOTENV_BAT = "dotenv.bat"

# The generated unset script, sourced right after the dotenv loader so an explicit clear beats a
# value `.env` filled in. pixi's own `[activation].env` is a string-to-string map with no way to
# say "not set", so a clear has to be shell rather than a table entry.
_UNSET_SH = "unset.sh"
_UNSET_BAT = "unset.bat"


def cleared(env: dict[str, str | bool]) -> list[str]:
    """The variables `env` asks to have taken away, in declaration order.

    A `false` value is a clear rather than a setting. It cannot ride in pixi's own
    `[activation].env`, which is a string-to-string map, so it becomes shell in the generated
    unset script instead and every consumer of that activation gets it for free.
    """
    return [name for name, value in env.items() if value is False]


def rerooted(path: str) -> str:
    """A workspace-relative path as the manifest generated under `.mainboard/` must spell it.

    ``packages/lote`` resolves from there as ``../packages/lote``, the one shift every declared
    location needs, a dependency source, a task working directory, an activation script. An
    absolute path is already unambiguous and rides through, and an empty one names the
    workspace root itself.
    """
    if PurePosixPath(path).is_absolute() or PureWindowsPath(path).is_absolute():
        return path
    return f"../{path}" if path else ".."


def _platform_name(entry: Toml) -> str:
    """Pixi platform name carried by a bare string or named descriptor."""
    if isinstance(entry, str):
        return entry
    if isinstance(entry, dict) and isinstance(name := entry.get("platform"), str):
        return name
    return ""


def _table(value: Toml | None) -> dict[str, Toml]:
    """Concrete table held by a recursive TOML value, empty for every scalar shape."""
    return dict(value) if isinstance(value, dict) else {}


def _reroot_source(spec: Toml) -> Toml:
    """A single dep spec with a local ``path`` source shifted up out of the generated directory.

    A bare version string, or a table without ``path``, rides through untouched.
    """
    if isinstance(spec, dict) and isinstance(path := spec.get("path"), str):
        return {**spec, "path": rerooted(path)}
    return spec


def _reparent(value: Toml) -> Toml:
    """Reroot local path deps in the compiled tables, leaving everything else as is.

    Only a ``path`` carried as a dependency *source* (a value under a ``dependencies`` or
    ``pypi-dependencies`` table) is shifted, so a dependency literally named ``path`` keeps its
    version untouched.
    """
    if isinstance(value, dict):
        return {
            key: {name: _reroot_source(spec) for name, spec in item.items()}
            if key in _DEP_TABLES and isinstance(item, dict)
            else _reparent(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_reparent(item) for item in value]
    return value


def spec_toml(spec: Spec) -> Toml:
    """The smallest TOML representation of one dependency spec.

    A bare version string when nothing else was declared, otherwise a table carrying the
    version (dropped when it is the unconstrained default) beside every extra key (`path`,
    `editable`, `git`, ...).
    """
    extra = spec.model_extra or {}
    if not extra:
        return spec.version
    named = {"version": spec.version} if spec.version != "*" else {}
    return {**named, **extra}


def _layered(declared: Mapping[str, Spec], *, over: Mapping[str, Spec]) -> dict[str, Spec]:
    """Each requirement _layered over the one it shadows, the way `Spec.merged` layers a scope.

    A pixi `[target]` dependency replaces its scope's own rather than narrowing it, so an
    overlay entry has to arrive already carrying whatever the scope said about that package.
    Otherwise `python = "*"` beside a declared `python = ">=3.14.6,<3.15"` reads as "python is
    present on this platform too" and compiles to "any python at all here", which is how a floor
    a workspace states once stops reaching one platform and lets that target's solve drift onto
    an interpreter nobody asked for.

    declared: the overlay's own requirements.
    over: the requirements it shadows, empty for a scope that overlays nothing.
    """
    return {
        name: spec.merged(over[name]) if name in over else spec for name, spec in declared.items()
    }


def dependency_tables(scope: Scope, *, over: Scope | None = None) -> dict[str, Toml]:
    """Compile one scope's conda and Python dependencies into Pixi dependency tables.

    Only the `python` ecosystem is understood beyond conda deps, mainboard's own
    simplification, python+conda through pixi for now, every other declared ecosystem table
    (`nodejs`, `rust`, ...) rides in the manifest but is not yet translated.

    over: the scope this one overlays, when it compiles into a `[target]` table rather than
        standing on its own. See `_layered` for why an overlay must not be compiled bare.
    """
    shadowed = over.deps if over else {}
    merged = _layered(scope.deps, over=shadowed)
    dependencies = {name: spec_toml(spec) for name, spec in merged.items()}
    tables: dict[str, Toml] = {"dependencies": dependencies} if dependencies else {}
    python: Toolchain | None = scope.toolchains().get("python")
    if python and python.all_deps():
        inherited: Toolchain | None = over.toolchains().get("python") if over else None
        requirements = _layered(python.all_deps(), over=inherited.all_deps() if inherited else {})
        tables["pypi-dependencies"] = {
            name: spec_toml(spec) for name, spec in requirements.items()
        }
    return tables


def pypi_options(scope: Scope) -> dict[str, Toml]:
    """Compile one scope's `[python]` solver settings into pixi's `[pypi-options]`.

    The settings sit beside `[python.deps]` as untyped extras, so the manifest never lags a uv
    feature, and only the keys pixi itself defines are forwarded. `dependency-overrides` is a
    dependency table like any other, so a local `path` in one is rerooted with the rest of them.
    """
    python: Toolchain | None = scope.toolchains().get("python")
    declared = (python.model_extra or {}) if python else {}
    return {key: declared[key] for key in _PYPI_OPTION_KEYS if key in declared}


class PixiManifest(FrozenModel):
    """The compiled pixi manifest (`pixi.toml`) emitted into the generated env."""

    workspace: dict[str, Toml]
    activation: dict[str, Toml] = {}
    dependencies: dict[str, Toml] = {}
    pypi_dependencies: dict[str, Toml] = Field(default_factory=dict, alias="pypi-dependencies")
    pypi_options: dict[str, Toml] = Field(default_factory=dict, alias="pypi-options")
    target: dict[str, Toml] = {}
    feature: dict[str, Toml] = {}
    environments: dict[str, Toml] = {}
    tasks: dict[str, Toml] = {}

    @staticmethod
    def activation_table(m: Manifest, *, windows: bool = False) -> dict[str, Toml]:
        """The `[activation]` table, exported env vars and the scripts pixi sources on entry.

        The generated dotenv loader lives beside the manifest and is sourced first, so a
        variable it loads is already visible to the rest of pixi's own activation and to every
        script the workspace declares after it. Those declared scripts are workspace-relative,
        the way everything else in the manifest is written, and are rerooted like any other
        declared location.
        """
        scripts: list[Toml] = [
            *([_DOTENV_BAT if windows else _DOTENV_SH] if m.workspace.dotenv else []),
            *([_UNSET_BAT if windows else _UNSET_SH] if cleared(m.env) else []),
            *(rerooted(script) for script in m.workspace.scripts),
        ]
        exported = {name: value for name, value in m.env.items() if isinstance(value, str)}
        return {
            **({"env": exported} if exported else {}),
            **({"scripts": scripts} if scripts else {}),
        }

    @staticmethod
    def platform_array(platforms: Iterable[Toml]) -> tomlkit.items.Array:
        """The workspace platform list as tomlkit items, each named variant an inline table."""
        rendered = tomlkit.array()
        for platform in platforms:
            if isinstance(platform, dict):
                descriptor = tomlkit.inline_table()
                descriptor.update(platform)
                rendered.append(descriptor)
            else:
                rendered.append(platform)
        return rendered

    @staticmethod
    def task(spec: Task) -> Toml:
        """Translate a manifest task into pixi's (`run` -> `cmd`, `depends` -> `depends-on`).

        A task that runs a command runs it from the repo root, one directory up from the
        generated `.mainboard/`, so repo-relative commands (`python -m pkg`) resolve as
        written, and a `dir` rebases that root. A command-less aggregator (only `depends`)
        carries no working directory, pixi rejects `cwd` without a `cmd`, so the rebase is
        skipped for it.
        """
        out: dict[str, Toml] = {}
        if isinstance(spec, str):
            out["cmd"] = spec
        else:
            renamed = {"run": "cmd", "depends": "depends-on", "dir": "cwd"}
            out = {renamed.get(key, key): value for key, value in spec.items()}
        if "cmd" in out:
            out["cwd"] = rerooted(str(out.get("cwd", "")))
        return out

    @classmethod
    def declared_feature(
        cls,
        name: str,
        env: Env,
        platforms: PlatformMatrix,
        *,
        clearing: bool = False,
        windows: bool = False,
    ) -> Toml:
        """One `[feature.<name>]` table: the env's own feature table, platforms and tasks.

        An environment declaring `no-default` starts from nothing but itself, and pixi reads
        that exclusion as covering the workspace `[activation]` table too, not only the deps.
        A clear is not a setting though. It is the workspace saying a variable must not be
        present at all, which an isolated environment has more reason to honour rather than
        less, and a variable the calling shell exported reaches an activated command whatever
        the environment solved from. So the generated unset script is carried into an isolated
        feature's own activation, and the environments that do include the default feature
        already source it there and are left alone.

        clearing: whether the workspace `[env]` table takes any variable away at all.
        windows: whether this feature can run on a Windows target.
        """
        body: dict[str, Toml] = {
            **cls.feature_table(env),
            **(
                {_PLATFORMS: platforms.environments[name]}
                if name in platforms.environments
                else {}
            ),
            **(
                {"tasks": {task: cls.task(spec) for task, spec in env.tasks.items()}}
                if env.tasks
                else {}
            ),
            **({"activation": {"scripts": [_UNSET_SH]}} if clearing and env.no_default else {}),
        }
        if not (clearing and env.no_default and windows):
            return body
        targets = _table(body.get("target"))
        windows_target = _table(targets.get("win"))
        targets["win"] = {**windows_target, "activation": {"scripts": [_UNSET_BAT]}}
        body["target"] = targets
        return body

    @classmethod
    def feature_table(cls, env: Env) -> dict[str, Toml]:
        """One env's own deps, channels, platforms and per-platform `[target]` overlays."""
        body = dependency_tables(env)
        if options := pypi_options(env):
            body[_PYPI_OPTIONS] = options
        if env.channels:
            body["channels"] = env.channels
        if env.platforms:
            body[_PLATFORMS] = env.platforms
        target = {
            platform: tables
            for platform, scope in env.on.items()
            if (tables := dependency_tables(scope, over=env))
        }
        if target:
            body["target"] = target
        return body

    @classmethod
    def features(
        cls, m: Manifest, platforms: PlatformMatrix, project_name: str
    ) -> tuple[dict[str, Toml], dict[str, Toml]]:
        """The `[feature]` and `[environments]` tables: one feature per declared env, plus ours.

        Beyond the declared envs, the tool owns two synthetic features that the default
        environment picks up: `<tool>-platforms` carries the workspace platform matrix, and
        `dev` carries the `[dev.*]` deps so a provisioned default env installs dev tooling
        beside the runtime deps.
        """
        clearing = bool(cleared(m.env))
        workspace_platforms = tuple(_platform_name(entry) for entry in platforms.workspace)
        feature: dict[str, Toml] = {
            name: cls.declared_feature(
                name,
                env,
                platforms,
                clearing=clearing,
                windows=any(
                    platform.startswith("win-")
                    for platform in (env.platforms or workspace_platforms)
                ),
            )
            for name, env in m.envs.items()
        }
        environments: dict[str, Toml] = {
            name: {
                "features": [name],
                **({"no-default-feature": True} if env.no_default else {}),
            }
            for name, env in m.envs.items()
        }
        owned: dict[str, Toml] = {
            **(
                {f"{project_name}-platforms": {_PLATFORMS: platforms.default}}
                if platforms.default
                else {}
            ),
            **({"dev": dev} if (dev := dependency_tables(m.dev)) else {}),
        }
        feature.update(owned)
        if owned:
            environments["default"] = {"features": list(owned)}
        return feature, environments

    @classmethod
    def from_manifest(cls, m: Manifest, *, project_name: str) -> Self:
        """Build the pixi manifest from a validated mainboard `Manifest`.

        `hosts`, `containers` and `vars` belong to other subsystems (remote dispatch, base
        images, template interpolation) and are never read here.

        project_name: the tool's own name (`Project().name`), naming the synthetic
            `<project_name>-platforms` feature so a rename never hardcodes it.
        """
        platforms = PlatformMatrix.from_manifest(m)
        feature, environments = cls.features(m, platforms, project_name)
        targets: dict[str, Toml] = {
            plat: dependency_tables(scope, over=m) for plat, scope in m.on.items()
        }
        if any(_platform_name(entry).startswith("win-") for entry in platforms.workspace) and (
            windows_activation := cls.activation_table(m, windows=True)
        ):
            targets["win"] = {**_table(targets.get("win")), "activation": windows_activation}
        payload: dict[str, Toml] = {
            "workspace": {
                "name": m.workspace.name,
                "version": m.workspace.version,
                "channels": m.workspace.channels,
                _PLATFORMS: platforms.workspace,
            },
            "activation": cls.activation_table(m),
            **dependency_tables(m),
            **({_PYPI_OPTIONS: options} if (options := pypi_options(m)) else {}),
            "target": targets,
            "feature": feature,
            "environments": environments,
            "tasks": {name: cls.task(spec) for name, spec in m.tasks.items()},
        }
        return cls.model_validate(_reparent(payload))

    def to_toml(self) -> str:
        """Render to `pixi.toml` text (hyphenated table names via the field aliases)."""
        body = self.model_dump(by_alias=True, exclude_defaults=True)
        body["workspace"][_PLATFORMS] = self.platform_array(body["workspace"][_PLATFORMS])
        return tomlkit.dumps(body)
