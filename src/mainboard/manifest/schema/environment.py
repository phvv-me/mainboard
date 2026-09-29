from pydantic import ConfigDict, Field

from ...core.base import kebab
from .scope import PlatformScope, Scope

# A bare command line, or a table of pixi's task keys (`run`, `dir`, `depends`) plus `env`.
type Task = str | dict[str, str | list[str] | dict[str, str]]


class Env(Scope):
    """A named environment: its own deps, overlays, tasks, and solve surface."""

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
