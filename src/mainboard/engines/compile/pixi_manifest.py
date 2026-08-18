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
    from collections.abc import Iterable

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
_DOTENV = "dotenv.sh"


def rerooted(path: str) -> str:
    """A workspace-relative path as the manifest generated under `.mainboard/` must spell it.

    ``packages/lote`` resolves from there as ``../packages/lote``, the one shift every declared
    location needs, a dependency source, a task working directory, an activation script. An
    absolute path is already unambiguous and rides through, and an empty one names the
    workspace root itself.
    """
    if path.startswith("/"):
        return path
    return f"../{path}" if path else ".."


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


def dependency_tables(scope: Scope) -> dict[str, Toml]:
    """Compile one scope's conda and Python dependencies into Pixi dependency tables.

    Only the `python` ecosystem is understood beyond conda deps, mainboard's own
    simplification, python+conda through pixi for now, every other declared ecosystem table
    (`nodejs`, `rust`, ...) rides in the manifest but is not yet translated.
    """
    dependencies = {name: spec_toml(spec) for name, spec in scope.deps.items()}
    tables: dict[str, Toml] = {"dependencies": dependencies} if dependencies else {}
    python: Toolchain | None = scope.toolchains().get("python")
    if python and python.all_deps():
        tables["pypi-dependencies"] = {
            name: spec_toml(spec) for name, spec in python.all_deps().items()
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
    pypi_options: dict[str, Toml] = Field(default_factory=dict, alias=_PYPI_OPTIONS)
    target: dict[str, Toml] = {}
    feature: dict[str, Toml] = {}
    environments: dict[str, Toml] = {}
    tasks: dict[str, Toml] = {}

    @staticmethod
    def activation_table(m: Manifest) -> dict[str, Toml]:
        """The `[activation]` table, exported env vars and the scripts pixi sources on entry.

        The generated dotenv loader lives beside the manifest and is sourced first, so a
        variable it loads is already visible to the rest of pixi's own activation and to every
        script the workspace declares after it. Those declared scripts are workspace-relative,
        the way everything else in the manifest is written, and are rerooted like any other
        declared location.
        """
        scripts: list[Toml] = [
            *([_DOTENV] if m.workspace.dotenv else []),
            *(rerooted(script) for script in m.workspace.scripts),
        ]
        return {
            **({"env": m.env} if m.env else {}),
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
            if (tables := dependency_tables(scope))
        }
        if target:
            body["target"] = target
        return body

    @classmethod
    def declared_feature(cls, name: str, env: Env, platforms: PlatformMatrix) -> Toml:
        """One `[feature.<name>]` table: the env's own feature table, platforms and tasks."""
        return {
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
        }

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
        feature: dict[str, Toml] = {
            name: cls.declared_feature(name, env, platforms) for name, env in m.envs.items()
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
            "target": {plat: dependency_tables(scope) for plat, scope in m.on.items()},
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
