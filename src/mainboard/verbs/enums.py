from enum import StrEnum, auto


class Accent(StrEnum):
    """The color a root group and everything under it print in for a person."""

    BLUE = auto()
    CYAN = auto()
    GREEN = auto()
    MAGENTA = auto()


class Scope(StrEnum):
    """Who besides the owner typing at a terminal may run a command; unset, it is no tool."""

    AGENT = auto()
    """an agent's tool too: it reads, or writes only what an agent may keep."""
    OWNER = auto()
    """the owner's act alone: it refuses in agent audience and is never a tool."""
    READ = auto()
    """a read-only tool: it changes nothing."""
