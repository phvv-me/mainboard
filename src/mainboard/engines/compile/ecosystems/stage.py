from collections.abc import Sequence
from typing import TYPE_CHECKING

from ....core.host import current_platform, platform_selectors
from ....manifest.schema.toolchain import Toolchain
from .base import Ecosystem

if TYPE_CHECKING:
    from pathlib import Path

    from ....manifest import Manifest, Scope
    from ....manifest.schema.environment import Env
    from ..backend import Pixi
    from ..generated import Writer


class SecondStage:
    """Every toolchain a workspace declares beyond conda and Python, for one manifest.

    pixi installs the conda environment, and the ecosystems here then fill it with whatever
    their own managers own. The three phases they take part in stay in step: the compile
    generates what those managers read, provisioning runs them, and activation exports the
    executables they linked. Which ecosystems take part is read from the manifest rather than
    listed here, so declaring `[go]` in a workspace that never had one needs no wiring.
    """

    def __init__(self, root: Path, manifest: Manifest, out: Path, pixi: Pixi) -> None:
        self.root = root
        self.manifest = manifest
        self.out = out
        self.pixi = pixi

    def binary_dirs(self, env: str) -> list[Path]:
        """Every directory the toolchains link executables into, for `env`'s activation."""
        return [directory for eco in self.ecosystems(env) for directory in eco.binary_dirs()]

    def ecosystems(self, env: str) -> list[Ecosystem]:
        """One bound ecosystem per implementation, in registration order.

        Every implementation is built, not only the ones the manifest still declares, because
        an ecosystem is also what cleans up after a table that was deleted: a `package.json`
        outliving its `[nodejs]` table would keep reinstalling packages nobody declares. A
        table no implementation claims (`[python]`, which pixi compiles itself) is ignored. An
        implementation whose tree is the workspace's rather than the environment's binds to the
        whole manifest instead (`Ecosystem.shared`).

        env: the environment whose merged tables the ecosystems bind to.
        """
        scoped = self.toolchains(env)
        shared = self.merged(self.shared_scopes())
        return [
            implementation(
                (shared if implementation.shared else scoped).get(
                    implementation.toolchain, Toolchain()
                ),
                env=env,
                project=self.manifest.workspace.name,
                workspace=self.root,
                out=self.out,
                pixi=self.pixi,
            )
            for implementation in Ecosystem.implementations()
        ]

    def generate(self, files: Writer, env: str) -> None:
        """Write every file the toolchains install from, under the sync lock the caller holds."""
        for ecosystem in self.ecosystems(env):
            ecosystem.generate(files)

    def install(self, env: str) -> None:
        """Run each toolchain's installer inside the environment pixi has already provisioned."""
        with self.pixi.activated(env):
            for ecosystem in self.ecosystems(env):
                ecosystem.sync()

    def merged(self, scopes: Sequence[Scope]) -> dict[str, Toolchain]:
        """Every ecosystem table across `scopes`, each merged over the ones beneath it."""
        merged: dict[str, Toolchain] = {}
        for scope in scopes:
            for name, table in scope.toolchains().items():
                merged[name] = table.merged(merged[name]) if name in merged else table
        return merged

    def overlays(self, scope: Manifest | Env) -> list[Scope]:
        """``scope`` followed by the platform overlays under it that this machine matches."""
        selectors = platform_selectors(current_platform())
        return [scope, *(over for key, over in scope.on.items() if key in selectors)]

    def scopes(self, env: str) -> list[Scope]:
        """Every scope whose tables apply to `env` on this machine, least specific first.

        The base manifest and its platform overlays come first (`[dev]` joins them for the
        default environment, the one a bare `mainboard run` uses), then the named environment
        and its own overlays, so a later table overrides an earlier one. An environment
        declaring `no-default` starts from nothing but itself, exactly as it solves in pixi.

        env: the environment name, refused here when the manifest never declared it.
        """
        named = self.manifest.environment(env)
        scopes: list[Scope] = []
        if not named.no_default:
            scopes.extend(self.overlays(self.manifest))
            if env == "default":
                scopes.append(self.manifest.dev)
        scopes.extend(self.overlays(named))
        return scopes

    def shared_scopes(self) -> list[Scope]:
        """Every scope this machine matches anywhere in the manifest, whichever env owns it.

        What a workspace-level toolchain installs cannot depend on which environment is being
        provisioned, because there is one generated `package.json` and one `node_modules` for
        all of them. Reading a single environment's view made provisioning an environment that
        declares no table of its own delete the file the others install from.
        """
        return [
            *self.overlays(self.manifest),
            self.manifest.dev,
            *(scope for env in self.manifest.envs.values() for scope in self.overlays(env)),
        ]

    def toolchains(self, env: str) -> dict[str, Toolchain]:
        """Every ecosystem table active for `env`, each merged over the scopes beneath it."""
        return self.merged(self.scopes(env))
