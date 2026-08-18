from pydantic import ConfigDict, Field

from ...core.base import kebab
from .scope import Scope

# A task is a bare command line, or a table of the keys pixi's own task takes: `run`, `dir`,
# `depends` and friends as strings and string lists, plus an `env` table of variables the task
# runs under.
type Task = str | dict[str, str | list[str] | dict[str, str]]


class Env(Scope):
    """A named environment: its own deps, overlays, tasks, and solve surface."""

    channels: list[str] = []
    on: dict[str, Scope] = {}
    tasks: dict[str, Task] = {}
    model_config = ConfigDict(alias_generator=kebab, populate_by_name=True)

    no_default: bool = Field(default=False)
    platforms: list[str] = []
    system: dict[str, str] = {}
