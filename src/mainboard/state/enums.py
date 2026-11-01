from enum import StrEnum, auto


class Access(StrEnum):
    """What an attach may do: read, write, create a lake that is not there, or migrate one."""

    READ = auto()
    WRITE = auto()
    CREATE = auto()
    MIGRATE = auto()
