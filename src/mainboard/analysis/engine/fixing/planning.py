# Planning one pass of the fixpoint: which fixes it takes together, and which it refuses.

from collections.abc import Mapping, Sequence, Set
from typing import TYPE_CHECKING

from ..apply import Plan
from ..checks import Syntax
from ..mirror import is_under

if TYPE_CHECKING:
    from pathlib import Path

    from ...catalog import Repair
    from ..facts import Facts
    from ..lint import Lint


class Planner:
    """Plans each pass of one fixpoint run.

    held: the files whose fixes are withheld, shared with the run; a file a pass would break
        joins it.
    refusals: why each fix that would remove a name another repository imports was refused.
    """

    def __init__(self, lint: Lint, applicability: tuple[str, ...], *, held: set[str]) -> None:
        self.lint, self.applicability, self.held = lint, applicability, held
        self.syntax = Syntax()
        self.refusals: dict[str, None] = {}

    def _refuse(self, groups: Sequence[Repair], lost: list[str], *, work: Path) -> set[int]:
        """The fixes among `groups` that cause the `lost` names.

        Each one alone, else every fix in the files they name. The reasons are kept in `refusals`.
        """
        named = {line.partition(":")[0] for line in lost}
        near = [group for group in groups if named & {edit.file() for edit in group}]
        alone = []
        for group in near:
            trial = Plan(work)
            trial.add(group)
            if self.lint.consumers.broken(trial):
                alone.append(group)
        self.refusals.update(dict.fromkeys(lost))
        return {id(group) for group in alone or near}

    def _taken(
        self, plan: Plan, repairs: Mapping[str, Sequence[Repair]], *, refused: Set[int]
    ) -> list[tuple[str, Repair]]:
        """The fixes `plan` takes, each with its rule, in rule order.

        A fix is taken unless refused, touching a withheld file, or overlapping one taken before.
        """
        taken = []
        for rule, candidates in repairs.items():
            for group in candidates:
                held = any(is_under(edit.file(), self.held) for edit in group)
                if id(group) not in refused and not held and plan.add(group):
                    taken.append((rule, group))
        return taken

    def plan(
        self, facts: Facts, *, work: Path
    ) -> tuple[Plan, list[tuple[str, Repair]], list[str]]:
        """One pass of every fix that overlaps no other and touches no withheld file.

        The rule of each fix taken is kept. A file the pass would leave unparsable is withheld and
        the pass planned again, as is a fix that removes a name another repository imports
        (consumers.py), for which only that fix is refused.
        """
        repairs = self.lint.repairs(facts, self.applicability)
        broken: list[str] = []
        refused: set[int] = set()
        while True:
            plan = Plan(work)
            groups = self._taken(plan, repairs, refused=refused)
            if newly := self.syntax.broken(plan):
                broken += newly
                self.held |= {line.partition(":")[0] for line in newly}
            elif lost := self.lint.consumers.broken(plan):
                refused |= self._refuse([group for _, group in groups], lost, work=work)
            else:
                return plan, groups, broken
