from functools import cached_property
from typing import TYPE_CHECKING, ClassVar

from patos import FrozenOpenModel

from ....core import MissionError, Project
from ..backend import Tool
from ..package_json import PackageJson
from .base import Ecosystem

if TYPE_CHECKING:
    from pathlib import Path

    from ....manifest.schema.spec import Json
    from ..generated import Writer

_MANIFEST = "package.json"
_MODULES = "node_modules"
_PACKAGE_FIELDS = "package"
_LOCKS = {
    "npm": ("npm-shrinkwrap.json", "package-lock.json"),
    "pnpm": ("pnpm-lock.yaml",),
    "yarn": ("yarn.lock",),
    "bun": ("bun.lock",),
}


class NodeOptions(FrozenOpenModel):
    """The `[nodejs]` settings beside its dependency tables.

    manager: the package manager binary that installs and links `node_modules`.
    app: whether this workspace is itself the JavaScript application, so its `package.json`
        and `node_modules` belong at the workspace root where a bundler resolves them.
    """

    manager: str = "npm"
    app: bool = False


class NodeManager(Tool):
    """The package manager `[nodejs] manager` names, run in the directory it installs into.

    npm, pnpm, yarn and bun share `package.json`, `node_modules` and install-into-cwd, so the
    binary name is all that differs.
    """

    def __init__(self, name: str, directory: Path) -> None:
        self.name = name
        self.directory = directory

    def available(self) -> bool:
        """Whether a `package.json` was generated to install from."""
        return (self.directory / _MANIFEST).is_file()

    def cwd(self) -> Path:
        return self.directory


class Node(Ecosystem):
    """The Node.js toolchain: a generated `package.json` plus the manager's lock.

    Both live in the generated directory, or at the workspace root with `app = true`, where a
    bundler and `node` resolve imports the way the ecosystem expects.
    """

    toolchain: ClassVar[str] = "nodejs"
    shared: ClassVar[bool] = True

    @property
    def directory(self) -> Path:
        return self.workspace if self.options.app else self.out

    @property
    def fields(self) -> dict[str, Json]:
        """The `[nodejs.package]` entries, merged verbatim into the generated manifest."""
        declared = (self.table.model_extra or {}).get(_PACKAGE_FIELDS)
        return dict(declared) if isinstance(declared, dict) else {}

    @property
    def manifest(self) -> Path:
        return self.directory / _MANIFEST

    @cached_property
    def options(self) -> NodeOptions:
        return NodeOptions.model_validate(self.table.model_extra or {})

    def binary_dirs(self) -> tuple[Path, ...]:
        return (self.directory / _MODULES / ".bin",)

    def compiled(self) -> PackageJson:
        """This table as a `package.json`, named `-npm` unless it is the application.

        The suffix keeps the generated manifest from claiming the workspace's published package.
        """
        name = self.project if self.options.app else f"{self.project}-npm"
        return PackageJson.compiled(
            name=name, deps=self.table.deps, dev=self.table.dev, fields=self.fields
        )

    def generate(self, files: Writer) -> None:
        """Write this table's `package.json`, or delete it (not empty it) once nothing is declared.

        A surviving one would keep reinstalling the last dependency removed.
        """
        if not (self.deps or self.fields):
            files.remove(self.manifest)
            return
        files.write(self.manifest, self.compiled().to_json())

    def frozen_inputs(self) -> tuple[Path, ...]:
        """Require a shard-local manifest and native lock before remote transfer or pinning."""
        if not (self.deps or self.fields):
            return ()
        if self.options.app:
            raise MissionError(
                "[nodejs] app=true installs into the mutable workspace, not an isolated "
                "prefix; frozen remote setup/dispatch for this mode is not implemented"
            )
        return self.manifest, self.lock()

    def lock(self) -> Path:
        """The selected manager's existing lock, preserving npm shrinkwrap precedence."""
        manager = self.options.manager
        try:
            names = _LOCKS[manager]
        except KeyError as missing:
            raise MissionError(
                f"[nodejs] manager={manager!r} has no supported frozen mode"
            ) from missing
        for name in names:
            path = self.directory / name
            if path.is_file():
                return path
        raise MissionError(
            f"{self.directory} has no {manager} lock ({', '.join(names)}); "
            f"run `{Project().name} install {self.env} --resolve` locally before shipping"
        )

    def sync(self, *, resolve: bool = False) -> None:
        """Install the generated manifest from its lock unless resolution was requested.

        The manager is a conda package on the activated environment's PATH and needs no
        environment flag, since its working directory is what it installs into.
        """
        if not (self.deps or self.fields):
            return
        manager = NodeManager(self.options.manager, self.directory)
        if not manager.available():
            raise MissionError(
                f"{self.manifest} is missing despite declared Node dependencies/package fields; "
                f"run `{Project().name} install {self.env}` to regenerate it"
            )
        if resolve:
            manager("install")
            return
        self.lock()
        if self.options.manager == "npm":
            manager("ci")
        else:
            manager("install", "--frozen-lockfile")
