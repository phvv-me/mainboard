import re
from pathlib import PurePath, PurePosixPath, PureWindowsPath
from typing import TYPE_CHECKING, Self

import tomlkit
import tomlkit.items
from patos import FrozenModel
from pydantic import Field

from ...core.host import platform_selectors
from .platforms import PlatformMatrix

# `Toml` backs pydantic fields below, so it must resolve at class-creation time.
from .toml import Toml
from .vendor import relocated

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Mapping

    from ...manifest import Env, Manifest, PlatformScope, Scope, Spec, Toolchain
    from ...manifest.schema.environment import Task

# pixi tables whose values are dependency specs, where a `path` source lives.
_DEP_TABLES = ("dependencies", "pypi-dependencies", "dependency-overrides")

_PLATFORMS = "platforms"
_PYPI_OPTIONS = "pypi-options"
_DEFAULT_GENERATED_DIR = PurePosixPath(".mainboard")

# One relative path token (leading parents and the segments under them), guarded against biting
# into a longer path (`/opt/a/../b`) or prose (`...`).
_RELATIVE = re.compile(r"(?<![\w./+~@-])\.\.(?:/[\w.+~@-]+)*(?![\w./+~@-])")

# pixi's own `[pypi-options]` fields; other keys beside `[python.deps]` (chefe's `indexes`)
# are not pixi's. `dependency-overrides` is last, the one sub-table, since TOML reads every key
# after a sub-table header as belonging to it.
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
_DOTENV_SH = "dotenv.sh"
_DOTENV_BAT = "dotenv.bat"

# The generated unset script, sourced after the dotenv loader so a clear beats `.env`.
_UNSET_SH = "unset.sh"
_UNSET_BAT = "unset.bat"


def cleared(env: dict[str, str | bool]) -> list[str]:
    """The variables `env` declares `false`, in order.

    pixi's `[activation].env` is a string map that cannot say "not set", so a clear becomes the
    generated unset script.
    """
    return [name for name, value in env.items() if value is False]


def rerooted(path: str, *, generated_dir: PurePath = _DEFAULT_GENERATED_DIR) -> str:
    """A workspace-relative path as a generated Pixi manifest must spell it.

    The parents are counted from `generated_dir`'s depth, never by hand. An absolute path rides
    through, and an empty one names the workspace root.
    """
    if PurePosixPath(path).is_absolute() or PureWindowsPath(path).is_absolute():
        return path
    parents = ("..",) * len(generated_dir.parts)
    root = PurePosixPath(*parents)
    return (root / PurePosixPath(path)).as_posix() if path else root.as_posix()


def anchored(
    text: str, *, root: PurePath, generated_dir: PurePath = _DEFAULT_GENERATED_DIR
) -> str:
    """One generated file's text, with every workspace-relative spelling resolved against `root`.

    The inverse of `rerooted`, for a file copied away from where it was compiled (a prefix sits
    one directory deeper, so `"../../.."` meant the generated directory and pixi refused it).
    Textual, since the lock and dotenv loader spell the same locations. Every spelling landing
    inside the workspace is rewritten, not only the one `rerooted` writes: pixi 0.79 re-spelled
    `../../../.mainboard/vendor/atpx` as `../../vendor/atpx`, which escaped an exact-prefix match
    and left a Miyabi wave importing no torch (2026-09-06).

    generated_dir: the directory it was compiled into, whose depth decides which relative
        spellings reach the workspace.
    """
    return _rewritten(
        text,
        generated_dir=generated_dir,
        spell=lambda inside: (root / inside).as_posix() if inside else root.as_posix(),
    )


def normalized(
    text: str, *, root: PurePath, generated_dir: PurePath = _DEFAULT_GENERATED_DIR
) -> str:
    """One generated file's text with every workspace location spelled one way, for a digest.

    pixi re-spells locations it was handed (as `pixi_lock.canonical` handles one level up), and
    `{{ config_root }}` renders each machine's own root into `[activation.env]`: the workstation
    pinned a4c06131efc5808c, the host read fc4975ef2096b9ac, and four Miyabi jobs died with no
    prefix built (2026-09-06). The root is written back out rather than activation stripped,
    so a real change to what a workspace exports still moves the address.

    root: the workspace root the file was compiled for, the one machine-specific spelling.
    """
    spelled = _unrooted(text, root=root, generated_dir=generated_dir)
    return _rewritten(
        spelled,
        generated_dir=generated_dir,
        spell=lambda inside: rerooted(inside, generated_dir=generated_dir),
    )


def _unrooted(text: str, *, root: PurePath, generated_dir: PurePath) -> str:
    """`text` with every absolute location under `root` written the way `rerooted` writes one."""
    here = re.escape(root.as_posix())
    anchored_at = re.compile(rf"(?<![\w./+~@-]){here}((?:/[\w.+~@-]+)*)(?![\w./+~@-])")
    return anchored_at.sub(
        lambda match: rerooted(match[1].lstrip("/"), generated_dir=generated_dir), text
    )


def _rewritten(text: str, *, generated_dir: PurePath, spell: Callable[[str], str]) -> str:
    """Every relative path token in `text` that reaches inside the workspace, respelled.

    A token climbing past the root is left as it stands: no mirror carries what it names.
    """

    def replace(match: re.Match[str]) -> str:
        inside = _inside(match[0], generated_dir=generated_dir)
        return match[0] if inside is None else spell(inside)

    return _RELATIVE.sub(replace, text)


def _inside(token: str, *, generated_dir: PurePath) -> str | None:
    """Where a relative token resolves under the workspace root (`""` for the root), else None.

    Pure arithmetic over the spelling; pathlib already drops every `.` component.
    """
    parts: list[str] = []
    for part in (*generated_dir.parts, *PurePosixPath(token).parts):
        if part != "..":
            parts.append(part)
        elif parts:
            parts.pop()
        else:
            return None
    return PurePosixPath(*parts).as_posix() if parts else ""


def self_installed(
    manifest: str, *, generated_dir: PurePath = _DEFAULT_GENERATED_DIR
) -> list[str]:
    """Every workspace-relative directory a compiled manifest installs as an editable package.

    An editable's directory decides what a job imports, as when a workspace installs its own
    root (`path = "."`). Read from the compiled manifest the prefix is built and digested from,
    so the two rosters never differ. Paths outside the workspace never travel and are left out.

    manifest: the compiled `pixi.toml`'s text.
    """
    parents = rerooted("", generated_dir=generated_dir)
    declared = [
        path
        for spec in _editable_specs(tomlkit.parse(manifest).unwrap())
        if isinstance(path := spec.get("path"), str)
    ]
    inside = [path for path in declared if path == parents or path.startswith(f"{parents}/")]
    return list(dict.fromkeys(path.removeprefix(parents).lstrip("/") for path in inside))


def _editable_specs(value: Toml) -> Iterator[dict[str, Toml]]:
    """Every editable dependency spec in the workspace, feature and platform-target tables."""
    if isinstance(value, dict):
        for key, item in value.items():
            if key in _DEP_TABLES and isinstance(item, dict):
                yield from (
                    spec
                    for spec in item.values()
                    if isinstance(spec, dict) and spec.get("editable")
                )
            else:
                yield from _editable_specs(item)
    elif isinstance(value, list):
        for item in value:
            yield from _editable_specs(item)


def _platform_name(entry: Toml) -> str:
    """The Pixi platform name a bare string or named descriptor carries."""
    if isinstance(entry, str):
        return entry
    if isinstance(entry, dict):
        for field in ("name", "platform"):
            if isinstance(name := entry.get(field), str):
                return name
    return ""


def _table(value: Toml | None) -> dict[str, Toml]:
    """The table a TOML value holds, empty for every other shape."""
    return dict(value) if isinstance(value, dict) else {}


def _reroot_source(name: str, spec: Toml, *, generated_dir: PurePath) -> Toml:
    """A dep spec with its local `path` source rerooted, at its vendored location if outside.

    See `vendor` for why a path leaving the root resolves nowhere on a host.
    """
    if isinstance(spec, dict) and isinstance(path := spec.get("path"), str):
        return {**spec, "path": rerooted(relocated(name, path), generated_dir=generated_dir)}
    return spec


def _reparent(value: Toml, *, generated_dir: PurePath) -> Toml:
    """Reroot the `path` sources in dependency tables, so a dependency named `path` is kept."""
    if isinstance(value, dict):
        return {
            key: {
                name: _reroot_source(name, spec, generated_dir=generated_dir)
                for name, spec in item.items()
            }
            if key in _DEP_TABLES and isinstance(item, dict)
            else _reparent(item, generated_dir=generated_dir)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_reparent(item, generated_dir=generated_dir) for item in value]
    return value


def spec_toml(spec: Spec) -> Toml:
    """The smallest TOML for one spec: a bare version, else a table without a `*` version."""
    extra = spec.model_extra or {}
    if not extra:
        return spec.version
    named = {"version": spec.version} if spec.version != "*" else {}
    return {**named, **extra}


def _layered(declared: Mapping[str, Spec], *, over: Mapping[str, Spec]) -> dict[str, Spec]:
    """Each requirement layered over the one it shadows, the way `Spec.merged` layers a scope.

    A pixi `[target]` dependency replaces its scope's own, so an overlay `python = "*"` would
    otherwise drop a declared `>=3.14.6,<3.15` floor on that platform.
    """
    return {
        name: spec.merged(over[name]) if name in over else spec for name, spec in declared.items()
    }


def dependency_tables(scope: Scope, *, over: Scope | None = None) -> dict[str, Toml]:
    """Compile one scope's conda and Python dependencies; other ecosystems are the second stage.

    over: the scope this one overlays when it compiles into a `[target]` table (see `_layered`).
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
    """Forward the pixi-defined `[python]` extras (untyped, never lagging uv) as pypi-options."""
    python: Toolchain | None = scope.toolchains().get("python")
    declared = (python.model_extra or {}) if python else {}
    return {key: declared[key] for key in _PYPI_OPTION_KEYS if key in declared}


def selected_manifest(manifest: Manifest, environment: str) -> Manifest:
    """The compile-visible manifest projection for one logical environment.

    `default` is root plus dev, a named environment root plus itself, a `no-default` one itself
    alone; unrelated environments never enter the shard. Non-compile tables are dropped, since a
    host naming a removed environment would fail referential validation.
    """
    selected = manifest.environment(environment)
    body = manifest.model_dump(mode="python", round_trip=True)
    body["envs"] = {} if environment == "default" else {environment: body["envs"][environment]}
    for field in manifest.uncompiled:
        body.pop(field, None)
    if environment != "default":
        body["dev"] = {}
    if selected.no_default:
        body["deps"] = {}
        body["on"] = {}
        body["system"] = {}
        for toolchain in manifest.toolchains():
            body.pop(toolchain, None)
    return type(manifest).model_validate(body)


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
    def platform_activation(scope: PlatformScope) -> dict[str, Toml]:
        """Environment values exported only while Pixi selects this target scope."""
        return {"env": dict(scope.env)} if scope.env else {}

    @classmethod
    def platform_target(cls, scope: PlatformScope, *, over: Scope) -> dict[str, Toml]:
        """One `[target.<platform>]` table: the overlay's dependencies and activation."""
        activation = cls.platform_activation(scope)
        return {
            **dependency_tables(scope, over=over),
            **({"activation": activation} if activation else {}),
        }

    @staticmethod
    def activation_table(
        m: Manifest,
        *,
        windows: bool = False,
        generated_dir: PurePath = _DEFAULT_GENERATED_DIR,
    ) -> dict[str, Toml]:
        """The `[activation]` table: exported env vars and the scripts pixi sources on entry.

        The dotenv loader comes first so every later script sees what it loads; declared
        scripts are workspace-relative and rerooted.
        """
        scripts: list[Toml] = [
            *([_DOTENV_BAT if windows else _DOTENV_SH] if m.workspace.dotenv else []),
            *([_UNSET_BAT if windows else _UNSET_SH] if cleared(m.env) else []),
            *(rerooted(script, generated_dir=generated_dir) for script in m.workspace.scripts),
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
    def task(spec: Task, *, generated_dir: PurePath = _DEFAULT_GENERATED_DIR) -> Toml:
        """Translate a manifest task into pixi's (`run` -> `cmd`, `depends` -> `depends-on`).

        A command runs from the repo root, rebased by `dir`. An aggregator gets no `cwd`, which
        pixi rejects without a `cmd`.
        """
        out: dict[str, Toml] = {}
        if isinstance(spec, str):
            out["cmd"] = spec
        else:
            renamed = {"run": "cmd", "depends": "depends-on", "dir": "cwd"}
            out = {renamed.get(key, key): value for key, value in spec.items()}
        if "cmd" in out:
            out["cwd"] = rerooted(str(out.get("cwd", "")), generated_dir=generated_dir)
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
        generated_dir: PurePath = _DEFAULT_GENERATED_DIR,
    ) -> Toml:
        """One `[feature.<name>]` table: the env's own feature table, platforms and tasks.

        pixi's `no-default` also drops the workspace `[activation]`, but a clear must still
        hold, so an isolated feature carries the unset script in its own activation.

        clearing: whether the workspace `[env]` table takes any variable away.
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
                {
                    "tasks": {
                        task: cls.task(spec, generated_dir=generated_dir)
                        for task, spec in env.tasks.items()
                    }
                }
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
            platform: table
            for platform, scope in env.on.items()
            if (table := cls.platform_target(scope, over=env))
        }
        if target:
            body["target"] = target
        return body

    @classmethod
    def features(
        cls,
        m: Manifest,
        platforms: PlatformMatrix,
        project_name: str,
        *,
        environment: str = "default",
        generated_dir: PurePath = _DEFAULT_GENERATED_DIR,
    ) -> tuple[dict[str, Toml], dict[str, Toml]]:
        """The features and sole logical environment carried by one Pixi shard.

        The default shard owns the synthetic dev and platform-routing features, a named shard
        only its own (Pixi layers the root feature unless `no-default`).
        """
        clearing = bool(cleared(m.env))
        workspace_platforms = tuple(
            _platform_name(entry) for entry in cls.workspace_platforms(platforms, environment)
        )
        if environment != "default":
            env = m.envs[environment]
            return (
                {
                    environment: cls.declared_feature(
                        environment,
                        env,
                        platforms,
                        clearing=clearing,
                        windows=any(
                            platform.startswith("win-")
                            for platform in (env.platforms or workspace_platforms)
                        ),
                        generated_dir=generated_dir,
                    )
                },
                {
                    environment: {
                        "features": [environment],
                        **({"no-default-feature": True} if env.no_default else {}),
                    }
                },
            )

        owned: dict[str, Toml] = {
            **(
                {f"{project_name}-platforms": {_PLATFORMS: platforms.default}}
                if platforms.default
                else {}
            ),
            **({"dev": dev} if (dev := dependency_tables(m.dev)) else {}),
        }
        return owned, ({"default": {"features": list(owned)}} if owned else {})

    @staticmethod
    def workspace_platforms(platforms: PlatformMatrix, environment: str) -> list[Toml]:
        """Only the platform descriptors the selected environment can solve.

        Without an explicit route the matrix already holds only the relevant bare entries.
        """
        names = (
            platforms.default
            if environment == "default" and platforms.default
            else platforms.environments.get(environment, [])
        )
        if not names:
            return platforms.workspace
        selected = set(names)
        return [entry for entry in platforms.workspace if _platform_name(entry) in selected]

    @classmethod
    def from_manifest(
        cls,
        m: Manifest,
        *,
        project_name: str,
        environment: str = "default",
        generated_dir: PurePath = _DEFAULT_GENERATED_DIR,
    ) -> Self:
        """Build one environment shard's Pixi manifest from a validated Mainboard manifest.

        project_name: names the synthetic `<project_name>-platforms` feature.
        generated_dir: workspace-relative directory containing the generated manifest.
        """
        m = selected_manifest(m, environment)
        platforms = PlatformMatrix.from_manifest(m)
        workspace_platforms = cls.workspace_platforms(platforms, environment)
        selectors = {
            selector
            for entry in workspace_platforms
            for selector in platform_selectors(
                entry["platform"]
                if isinstance(entry, dict) and isinstance(entry.get("platform"), str)
                else _platform_name(entry)
            )
        }
        feature, environments = cls.features(
            m,
            platforms,
            project_name,
            environment=environment,
            generated_dir=generated_dir,
        )
        targets: dict[str, Toml] = {
            platform: cls.platform_target(scope, over=m)
            for platform, scope in m.on.items()
            if platform in selectors
        }
        for body in feature.values():
            if isinstance(body, dict) and isinstance(target := body.get("target"), dict):
                body["target"] = {
                    platform: scope for platform, scope in target.items() if platform in selectors
                }
        if any(_platform_name(entry).startswith("win-") for entry in workspace_platforms) and (
            windows_activation := cls.activation_table(
                m, windows=True, generated_dir=generated_dir
            )
        ):
            windows_target = _table(targets.get("win"))
            platform_activation = _table(windows_target.get("activation"))
            windows_environment = {
                **_table(windows_activation.get("env")),
                **_table(platform_activation.get("env")),
            }
            targets["win"] = {
                **windows_target,
                "activation": {
                    **windows_activation,
                    **({"env": windows_environment} if windows_environment else {}),
                },
            }
        payload: dict[str, Toml] = {
            "workspace": {
                "name": m.workspace.name,
                "version": m.workspace.version,
                "channels": m.workspace.channels,
                _PLATFORMS: workspace_platforms,
            },
            "activation": cls.activation_table(m, generated_dir=generated_dir),
            **dependency_tables(m),
            **({_PYPI_OPTIONS: options} if (options := pypi_options(m)) else {}),
            "target": targets,
            "feature": feature,
            "environments": environments,
            "tasks": {
                name: cls.task(spec, generated_dir=generated_dir) for name, spec in m.tasks.items()
            },
        }
        return cls.model_validate(_reparent(payload, generated_dir=generated_dir))

    def to_toml(self) -> str:
        """Render to `pixi.toml` text, hyphenated names via the field aliases."""
        body = self.model_dump(by_alias=True, exclude_defaults=True)
        body["workspace"][_PLATFORMS] = self.platform_array(body["workspace"][_PLATFORMS])
        return tomlkit.dumps(body)
