"""What every command prints: compact text, or JSON for machines."""

from abc import ABC, abstractmethod

from patos import FrozenModel


class Report(FrozenModel, ABC):
    """A command's answer, rendered as text for a reader or as JSON for a program."""

    @abstractmethod
    def text(self) -> str:
        """The compact rendering a terminal shows."""

    def render(self, *, as_json: bool) -> str:
        return self.model_dump_json(indent=2) if as_json else self.text()
