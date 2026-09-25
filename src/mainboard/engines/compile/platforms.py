from collections.abc import Sequence
from typing import TYPE_CHECKING, Self

from patos import FrozenModel

from ...core.host import current_platform, platform_family
from .toml import Toml

if TYPE_CHECKING:
    from ...manifest import Manifest

# Every platform family pixi names a virtual package for, the reach of an unlisted floor.
_EVERY_FAMILY = frozenset({"linux", "osx", "win"})

# The families whose machines carry each floor's virtual package: a macOS target means nothing
# to a Linux solve, glibc to a macOS one, CUDA to Apple. A key pixi adds later rides everywhere.
_FLOOR_FAMILIES: dict[str, frozenset[str]] = {
    "archspec": _EVERY_FAMILY,
    "cuda": frozenset({"linux", "win"}),
    "glibc": frozenset({"linux"}),
    "linux": frozenset({"linux"}),
    "macos": frozenset({"osx"}),
    "osx": frozenset({"osx"}),
    "windows": frozenset({"win"}),
}


class SystemFloors(FrozenModel):
    """One `[system]` table, answering which of its floors a given platform can meet.

    Copying `macos = "14.0"` onto a Linux platform makes pixi warn every Linux clone that the
    machine does not provide `__osx`.

    declared: keyed by pixi's virtual package name.
    """

    declared: dict[str, str] = {}

    def on(self, platform: str) -> dict[str, str]:
        """The declared floors that mean something on one pixi platform (`linux-aarch64`)."""
        family = platform_family(platform)
        return {
            key: value
            for key, value in self.declared.items()
            if family in _FLOOR_FAMILIES.get(key, _EVERY_FAMILY)
        }


class PlatformVariant(FrozenModel):
    """One entry in pixi's platform list: the bare platform, or with floors a named
    `<platform>-<suffix>` table each environment selects by name.

    suffix: `system` for the workspace's own floors, else the environment that raised them.
    floors: already scoped to the platform.
    """

    platform: str
    suffix: str
    floors: dict[str, str] = {}

    @property
    def name(self) -> str:
        """What a feature or environment spells to select this entry."""
        return f"{self.platform}-{self.suffix}" if self.floors else self.platform

    def descriptor(self) -> Toml:
        """This entry as pixi reads it, an inline table only when it raises a floor."""
        if not self.floors:
            return self.platform
        return {"name": self.name, "platform": self.platform, **self.floors}


class PlatformMatrix(FrozenModel):
    """Pixi platform descriptors and the feature routes that select them."""

    workspace: list[Toml]
    environments: dict[str, list[str]]
    default: list[str]

    @staticmethod
    def spread(
        system: dict[str, str], suffix: str, platforms: Sequence[str]
    ) -> dict[str, PlatformVariant]:
        """One `[system]` table over `platforms`, each keeping only the floors it can meet."""
        floors = SystemFloors(declared=system)
        return {
            platform: PlatformVariant(platform=platform, suffix=suffix, floors=floors.on(platform))
            for platform in platforms
        }

    @classmethod
    def from_manifest(cls, manifest: Manifest) -> Self:
        """Expand a manifest's virtual package floors into named Pixi platform variants.

        An env raising its own floors gets a `<platform>-<env>` variant wherever they reach and
        the workspace's entry elsewhere, and a floor surviving anywhere forces each env to name
        the variants it runs on.
        """
        # An undeclared platform list means this machine, so a zero-config manifest installs here.
        platforms = manifest.workspace.platforms or [current_platform()]
        root = cls.spread(manifest.system, "system", platforms)
        # Spread again rather than index `root`: an isolated environment may add a platform the
        # default environment cannot solve.
        chosen = {
            name: cls.spread(env.system, name, env.platforms or platforms)
            if env.system and env.system != manifest.system
            else cls.spread(manifest.system, "system", env.platforms or platforms)
            for name, env in manifest.envs.items()
        }
        picked = [variant for selection in chosen.values() for variant in selection.values()]
        entries = {variant.name: variant for variant in (*root.values(), *picked)}
        routed = any(variant.floors for variant in entries.values())
        return cls(
            workspace=[variant.descriptor() for variant in entries.values()],
            environments={
                name: [variant.name for variant in chosen[name].values()]
                for name, env in manifest.envs.items()
                if env.platforms or routed
            },
            default=[variant.name for variant in root.values()] if routed else [],
        )
