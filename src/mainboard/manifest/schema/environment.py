from typing import Self

from pydantic import ConfigDict, Field

from ...core.base import kebab
from .scope import PlatformScope, Scope

# A bare command line, or a table of pixi's task keys (`run`, `dir`, `depends`) plus `env`.
type Task = str | dict[str, str | list[str] | dict[str, str]]


class Env(Scope):
    """A named environment: its own deps, overlays, tasks, and solve surface.

    holds: the requirements of this environment `upgrade` leaves as written, each with the reason
        (`torch` beside a vLLM that pins it exactly), as the root's `[holds]` does everywhere.
    """

    channels: list[str] = []
    on: dict[str, PlatformScope] = {}
    tasks: dict[str, Task] = {}
    model_config = ConfigDict(alias_generator=kebab, populate_by_name=True)

    no_default: bool = Field(default=False)
    platforms: list[str] = []
    system: dict[str, str] = {}
    # The workspace paths this environment's code lives in. A host running it mirrors these
    # (with the manifest and lock) instead of its sync include: a rented machine running cutok
    # needs cutok's sources and the workspace packages it imports, not every project's.
    sources: list[str] = []
    holds: dict[str, str] = {}

    def merged(self, over: Self) -> Self:
        """This environment layered over `over`, another manifest's of the same name.

        Requirements, overlays, tasks, channels and sources are joined, and this one's settings and
        bounds win where both set them, so a project's contribution never loosens what the root
        declares.
        """
        merged = super().merged(over)
        on = {
            platform: self.on[platform].merged(over.on[platform])
            if platform in self.on and platform in over.on
            else self.on.get(platform) or over.on[platform]
            for platform in self.on.keys() | over.on.keys()
        }
        return merged.model_copy(
            update={
                "channels": list(dict.fromkeys([*self.channels, *over.channels])),
                "on": on,
                "tasks": over.tasks | self.tasks,
                "no_default": self.no_default or over.no_default,
                "platforms": self.platforms or over.platforms,
                "system": over.system | self.system,
                "sources": list(dict.fromkeys([*self.sources, *over.sources])),
                "holds": over.holds | self.holds,
            }
        )
