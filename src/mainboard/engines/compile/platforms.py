from typing import TYPE_CHECKING, Self

from patos import FrozenModel

from ...core.host import current_platform, platform_family
from .toml import Toml

if TYPE_CHECKING:
    from ...manifest import Manifest

# Every platform family pixi names a virtual package for, and the reach of any floor this table
# below does not pin down.
_EVERY_FAMILY = frozenset({"linux", "osx", "win"})

# Which families can actually provide each floor pixi accepts on a platform descriptor. A macOS
# deployment target is meaningless to a Linux solve, a glibc version to a macOS one, and CUDA to
# either kind of Apple machine, so a floor never travels outside the family whose machines carry
# the matching virtual package. `archspec` is a microarchitecture every machine has, and a key
# pixi learns after this table was written rides everywhere rather than being dropped.
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

    Declaring `macos = "14.0"` states a deployment target for a workspace's Apple builds, not a
    requirement its Linux builds have to satisfy, and copying it onto a Linux platform is what
    makes pixi warn every Linux clone that the machine does not provide `__osx`.

    declared: the floors as the manifest wrote them, keyed by pixi's virtual package name.
    """

    declared: dict[str, str] = {}

    def on(self, platform: str) -> dict[str, str]:
        """The declared floors that mean something on one pixi platform.

        platform: a pixi platform string such as `linux-aarch64`.
        """
        family = platform_family(platform)
        return {
            key: value
            for key, value in self.declared.items()
            if family in _FLOOR_FAMILIES.get(key, _EVERY_FAMILY)
        }


class PlatformVariant(FrozenModel):
    """One entry in pixi's workspace platform list, bare or named after the floors it carries.

    Floors are what give a platform a name at all: an entry with none is the plain platform
    string every pixi manifest already spells, and only an entry that genuinely raises something
    becomes a `<platform>-<suffix>` table each environment has to select by name.

    platform: the pixi platform this entry solves for.
    suffix: what the named form is called after, `system` for the workspace's own floors and the
        environment's name for a floor that environment raised.
    floors: the virtual package floors this entry carries, already scoped to the platform.
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
        system: dict[str, str], suffix: str, platforms: list[str]
    ) -> dict[str, PlatformVariant]:
        """One `[system]` table over `platforms`, each entry keeping only the floors it can meet.

        system: the declared floors, keyed by pixi's virtual package name.
        suffix: what the named variants are called after.
        platforms: the pixi platforms the table is spread across.
        """
        floors = SystemFloors(declared=system)
        return {
            platform: PlatformVariant(platform=platform, suffix=suffix, floors=floors.on(platform))
            for platform in platforms
        }

    @classmethod
    def from_manifest(cls, manifest: Manifest) -> Self:
        """Expand a manifest's virtual package floors into named Pixi platform variants.

        A floor reaches only the platforms whose family can provide it, so a macOS deployment
        target lands on the osx targets alone and a Linux collaborator never carries an `__osx`
        requirement no machine of theirs can satisfy. An env that raises its own floors gets a
        `<platform>-<env>` variant of every platform those floors reach and rides the workspace's
        own entry everywhere else, and a floor that survives anywhere forces each env to name the
        variants it runs on.
        """
        # An undeclared platform list means this machine, so a zero-config manifest
        # compiles to a workspace pixi can actually install here.
        platforms = manifest.workspace.platforms or [current_platform()]
        root = cls.spread(manifest.system, "system", platforms)
        chosen = {
            name: cls.spread(env.system, name, env.platforms or platforms)
            if env.system and env.system != manifest.system
            else {platform: root[platform] for platform in env.platforms or platforms}
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
