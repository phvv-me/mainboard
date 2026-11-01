from cyclopts import Parameter
from patos import FrozenModel
from pydantic import ConfigDict

from ....core.errors import MissionError
from ....enums import LockPolicy


@Parameter(name="*")
class Locking(FrozenModel):
    """What a verb may do to the lock before it installs, pixi's `--locked` and `--frozen`."""

    model_config = ConfigDict(use_attribute_docstrings=True)

    locked: bool = False
    """refuse a lock that no longer answers the manifest."""
    frozen: bool = False
    """take the lock as it stands, never comparing it with the manifest."""

    @property
    def policy(self) -> LockPolicy:
        """The policy these flags name, refusing both at once."""
        if self.locked and self.frozen:
            raise MissionError("--locked and --frozen exclude each other")
        if self.locked:
            return LockPolicy.LOCKED
        return LockPolicy.FROZEN if self.frozen else LockPolicy.UPDATE


@Parameter(name="*")
class Preparation(Locking):
    """What a verb may do to the lock and the environment before it runs, as pixi's flags say."""

    no_install: bool = False
    """leave the environment as it is installed."""
    as_is: bool = False
    """`--frozen` and `--no-install` together: run what is installed, as it is."""

    @property
    def installs(self) -> bool:
        """Whether the environment is brought in line with the lock first."""
        return not (self.no_install or self.as_is)

    @property
    def policy(self) -> LockPolicy:
        return Locking(locked=self.locked, frozen=self.frozen or self.as_is).policy
