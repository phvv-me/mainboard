from functools import cached_property
from typing import TYPE_CHECKING, ClassVar

from patos import FrozenOpenModel

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

    The manager is the whole of the backend, since `package.json` is the only thing it reads
    and `node_modules` the only thing it writes. An ordinary toolchain keeps both inside the
    generated environment directory, while `app = true` moves them to the workspace root,
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
        if not self.deps:
            files.remove(self.manifest)
            return
        files.write(self.manifest, self.compiled().to_json())

    def sync(self) -> None:
        """Install the generated `package.json` with the declared manager.

        The manager is itself a conda package, reached through the activated environment the
        second stage runs inside, and it needs no environment flag of its own because the
        directory it runs in is the environment it installs into.
        """
        NodeManager(self.options.manager, self.directory)("install")
