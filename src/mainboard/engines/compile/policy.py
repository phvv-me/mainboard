from enum import StrEnum, auto

from ...core.errors import MissionError


class LockPolicy(StrEnum):
    """What a verb may do to the lock before it installs, spelled as pixi spells it.

    update: solve again when the lock no longer answers the manifest, pixi's default.
    locked: refuse a lock that no longer answers the manifest (`--locked`), what a host does,
        since a host installs what the center solved and never solves for itself.
    frozen: take the lock as it stands without asking whether it answers (`--frozen`).
    """

    UPDATE = auto()
    LOCKED = auto()
    FROZEN = auto()

    @classmethod
    def of(cls, *, locked: bool = False, frozen: bool = False) -> LockPolicy:
        """The policy pixi's `--locked` and `--frozen` flags name, refusing both at once."""
        if locked and frozen:
            raise MissionError("--locked and --frozen exclude each other")
        return cls.LOCKED if locked else cls.FROZEN if frozen else cls.UPDATE

    @property
    def flag(self) -> str:
        """The flag passing this policy on to another machine's install, empty for the default."""
        return "" if self is LockPolicy.UPDATE else f"--{self}"
