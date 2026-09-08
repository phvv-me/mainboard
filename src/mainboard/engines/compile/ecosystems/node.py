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
    """The `[nodejs]` settings that sit beside its dependency tables.

    manager: the package manager binary that installs and links `node_modules`.
    app: whether this workspace is itself the JavaScript application, so its `package.json`
        and `node_modules` belong at the workspace root where a bundler resolves them.
    """

    manager: str = "npm"
    app: bool = False


class NodeManager(Tool):
    """The package manager `[nodejs] manager` names, run in the directory it installs into.

    npm, pnpm, yarn and bun read the same `package.json` and write the same `node_modules`,
    and each installs into its working directory rather than behind a per-tool flag, so a
    manifest naming a different manager needs nothing here but that manager's binary name.
    """

    def __init__(self, name: str, directory: Path) -> None:
        self.name = name
        self.directory = directory

    def available(self) -> bool:
        """Whether a `package.json` was generated for this manager to install from."""
        return (self.directory / _MANIFEST).is_file()

    def cwd(self) -> Path:
        return self.directory


class Node(Ecosystem):
    """The Node.js toolchain: a generated `package.json`, installed by the declared manager.

    The generated manifest and the manager's resolved lock jointly define the install.
    An ordinary toolchain keeps them inside the generated environment directory,
    while `app = true` moves them to the workspace root,
    where a bundler and a `node` process resolve imports the way the ecosystem expects.
    """

    toolchain: ClassVar[str] = "nodejs"
    # One `package.json` and one `node_modules` serve the whole workspace, in the generated
    # directory or at the root, never one per environment.
    shared: ClassVar[bool] = True

    @property
    def directory(self) -> Path:
        """Where `package.json` and `node_modules` live for this toolchain."""
        return self.workspace if self.options.app else self.out

    @property
    def fields(self) -> dict[str, Json]:
        """The `[nodejs.package]` entries, merged verbatim into the generated manifest."""
        declared = (self.table.model_extra or {}).get(_PACKAGE_FIELDS)
        return dict(declared) if isinstance(declared, dict) else {}

    @property
    def manifest(self) -> Path:
        """The generated `package.json` the manager installs from."""
        return self.directory / _MANIFEST

    @cached_property
    def options(self) -> NodeOptions:
        """The table's settings beyond its deps, defaulted when it declares none."""
        return NodeOptions.model_validate(self.table.model_extra or {})

    def binary_dirs(self) -> tuple[Path, ...]:
        """Where the manager links the executables its packages ship."""
        return (self.directory / _MODULES / ".bin",)

    def compiled(self) -> PackageJson:
        """This table as a `package.json`.

        A toolchain that is not the application gets a `-npm` suffixed name, so the generated
        manifest never claims to be the package the workspace itself publishes.
        """
        name = self.project if self.options.app else f"{self.project}-npm"
        return PackageJson.compiled(
            name=name, deps=self.table.deps, dev=self.table.dev, fields=self.fields
        )

    def generate(self, files: Writer) -> None:
        """Write the `package.json` for this table, or drop the one a bare table left behind.

        A `package.json` surviving the removal of the last declared dependency would keep
        reinstalling it, so the file is deleted rather than emptied.
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

        The manager is itself a conda package, reached through the activated environment the
        second stage runs inside, and it needs no environment flag of its own because the
        directory it runs in is the environment it installs into.
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
