# A structural operation a fix row names: SQL chooses what moves and gathers its context, and the
# operation renders the edits, so a rule states the change once whatever the source spells.

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from patos import Registry

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ..edits import Edit


class Operation(Registry, ABC):
    """One kind of structural change, found by the `operation` a fix row names (`move-constants`).

    An operation answers the edits that make its change, or None to refuse it: a refusal withholds
    every edit of the row's finding, so a change is made whole or not at all, and says why.
    """

    why = ""

    @abstractmethod
    def edits(self, row: Mapping) -> list[Edit] | None:
        """The edits this change makes from `row`'s columns, None when it cannot be made safely."""

    @classmethod
    def rendered(cls, row: Mapping) -> list[Edit] | None:
        """The edits the operation `row` names makes from it, None when it refuses."""
        return cls.find(row["operation"])().edits(row)

    def refused(self, why: str) -> None:
        """Refuse the change for the reason `why`."""
        self.why = why
