from typing import TYPE_CHECKING, Self

from patos import FrozenModel

from ...core.host import current_platform
from .toml import Toml

if TYPE_CHECKING:
    from ...manifest import Manifest


class PlatformMatrix(FrozenModel):
    """Pixi platform descriptors and the feature routes that select them."""

    workspace: list[Toml]
    environments: dict[str, list[str]]
    default: list[str]

    @staticmethod
    def descriptor(name: str, *, platform: str, system: dict[str, str]) -> Toml:
        """One workspace platform entry, a bare platform unless virtual package floors apply."""
        return {"name": name, "platform": platform, **system} if system else platform

    @classmethod
    def from_manifest(cls, manifest: Manifest) -> Self:
        """Expand a manifest's virtual package floors into named Pixi platform variants.

        An env that raises its own floors gets a `<platform>-<env>` variant of every platform it
        selects, and any floor anywhere forces each env to name the variants it runs on.
        """
        # An undeclared platform list means this machine, so a zero-config manifest
        # compiles to a workspace pixi can actually install here.
        platforms = manifest.workspace.platforms or [current_platform()]
        root = {p: f"{p}-system" if manifest.system else p for p in platforms}
        raised = {
            name: env.system
            for name, env in manifest.envs.items()
            if env.system and env.system != manifest.system
        }
        selected = {name: env.platforms or platforms for name, env in manifest.envs.items()}
        routed = bool(manifest.system or any(env.system for env in manifest.envs.values()))
        workspace: list[Toml] = [
            cls.descriptor(root[p], platform=p, system=manifest.system) for p in platforms
        ]
        workspace.extend(
            cls.descriptor(f"{p}-{name}", platform=p, system=system)
            for name, system in raised.items()
            for p in selected[name]
        )
        environments = {
            name: [f"{p}-{name}" if name in raised else root[p] for p in selected[name]]
            for name, env in manifest.envs.items()
            if env.platforms or routed
        }
        return cls(
            workspace=workspace,
            environments=environments,
            default=list(root.values()) if routed else [],
        )
