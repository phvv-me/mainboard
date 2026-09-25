import hashlib
import json
from collections.abc import Sequence
from pathlib import PurePosixPath
from typing import TYPE_CHECKING

from ....core.host import current_platform, platform_selectors
from ....manifest.schema.toolchain import Toolchain
from ..pixi_manifest import normalized
from .base import Ecosystem

if TYPE_CHECKING:
    from pathlib import Path

    from ....manifest import Manifest, Scope
    from ....manifest.schema.environment import Env
    from ..backend import Pixi
    from ..generated import Writer


class SecondStage:
    """Every toolchain a manifest declares beyond conda and Python.

    The compile generates what their managers read, provisioning runs them, and activation
    exports what they linked. Participation comes from the manifest, so a new `[go]` needs no
    wiring.
    """

    def __init__(self, root: Path, manifest: Manifest, out: Path, pixi: Pixi) -> None:
        self.root = root
        self.manifest = manifest
        self.out = out
        self.pixi = pixi

    def digest(self) -> str:
        """A location-independent digest of every second-stage table, in scope and overlay order.

        Rust, Go and Node's manager and app settings never reach a generated file, so the
        declarations themselves are hashed, leaving tasks and host tables out.
        """
        scopes = [
            ("root", self.manifest),
            ("dev", self.manifest.dev),
            *((f"on:{name}", scope) for name, scope in self.manifest.on.items()),
            *(
                item
                for name, environment in self.manifest.envs.items()
                for item in (
                    (f"env:{name}", environment),
                    *(
                        (f"env:{name}/on:{platform}", scope)
                        for platform, scope in environment.on.items()
                    ),
                )
            ),
        ]
        declared = {implementation.toolchain for implementation in Ecosystem.implementations()}
        payload = {
            "project": self.manifest.workspace.name,
            "scopes": [
                (
                    name,
                    {
                        key: table.model_dump(mode="json", round_trip=True)
                        for key, table in scope.toolchains().items()
                        if key in declared
                    },
                )
                for name, scope in scopes
            ],
        }
        text = normalized(
            json.dumps(payload, separators=(",", ":")),
            root=self.root,
            generated_dir=PurePosixPath("."),
        )
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def binary_dirs(self, env: str) -> list[Path]:
        """Every directory the toolchains link executables into, for `env`'s activation."""
        return [directory for eco in self.ecosystems(env) for directory in eco.binary_dirs()]

    def ecosystems(self, env: str) -> list[Ecosystem]:
        """One bound ecosystem per implementation, in registration order.

        Every implementation is built, declared or not, since it is what cleans up after a
        deleted table. A table none claims (`[python]`, pixi's own) is ignored, and a `shared`
        implementation binds to the whole manifest.
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

    def frozen_inputs(self, env: str) -> tuple[Path, ...]:
        """Require each selected runtime's native frozen contract before transferring files."""
        return tuple(path for eco in self.ecosystems(env) for path in eco.frozen_inputs())

    def install(self, env: str, *, resolve: bool = False) -> None:
        """Install locked inputs; only an explicit resolve may choose new package versions."""
        with self.pixi.activated(env):
            for ecosystem in self.ecosystems(env):
                ecosystem.sync(resolve=resolve)

    def merged(self, scopes: Sequence[Scope]) -> dict[str, Toolchain]:
        """Every ecosystem table across `scopes`, each merged over the ones before it."""
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

        The base manifest and its overlays (plus `[dev]` for `default`), then the named
        environment and its overlays. `no-default` starts from the environment alone, as pixi
        solves it. An undeclared `env` is refused.
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

        A shared toolchain has one tree for every environment, so reading one environment's view
        would let provisioning an environment with no table delete what the others install from.
        """
        return [
            *self.overlays(self.manifest),
            self.manifest.dev,
            *(scope for env in self.manifest.envs.values() for scope in self.overlays(env)),
        ]

    def toolchains(self, env: str) -> dict[str, Toolchain]:
        """Every ecosystem table active for `env`, each merged over the scopes beneath it."""
        return self.merged(self.scopes(env))
