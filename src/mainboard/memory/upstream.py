"""The memory behind `mb memory`: a remote that answers questions, takes keeps and says how it is.

It fails as `core.errors`' `Unreachable` (what it was sent can wait), `Refused` (it never takes
that) or `Unavailable` (this machine cannot use it). `outbox.Memory` is the one front every call
goes through; the client adapter imports this contract, and nothing it imports loads the adapter.
"""

from abc import ABC, abstractmethod

from patos import FrozenModel

from .keeps import Keep


class Remote(FrozenModel):
    """How the upstream says it is.

    login: whether this machine holds a session it can use.
    processing: what the upstream is doing with what it holds.
    """

    login: bool
    caller: str = ""
    finds: int = 0
    keeps: int = 0
    processing: str = ""


class Upstream(ABC):
    """A remote memory: every method raises `Unreachable` when it does not answer, and `Refused`
    when it answers no."""

    @abstractmethod
    def find(self, question: str, *, budget: int | None = None) -> str:
        """Answer `question` from memory, in at most `budget` tokens when given."""

    @abstractmethod
    def keep(self, keep: Keep) -> str:
        """Keep one item; answers the upstream's id for what it kept."""

    @abstractmethod
    def status(self) -> Remote:
        """How the upstream is."""

    @property
    @abstractmethod
    def server(self) -> str:
        """Where it answers."""
