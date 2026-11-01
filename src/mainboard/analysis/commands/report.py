"""What every command prints: compact text, or JSON for machines."""

from abc import ABC, abstractmethod

from patos import FrozenModel

from ...render import Output


class Report(FrozenModel, ABC):
    """A command's answer, shown as text for a reader or as JSON for a program."""

    @abstractmethod
    def text(self) -> str:
        """The compact rendering a terminal shows."""

    def show(self, output: Output) -> None:
        """Print the text, or the model as JSON under `--json`."""
        output.print_document(self.text(), self.model_dump(mode="json"), title=type(self).__name__)
