# One pass of fixes as a diff, before any proof: what the rules' fixes would write, to read.

import difflib
from pathlib import Path

from ....log import logger
from ...constants import FIXABLE
from ...engine.apply import Plan
from ...engine.lint.verbs import Selection

# Every rule, at the default budget.
_EVERY = Selection()


def preview(
    root: Path = Path(),
    *,
    selection: Selection = _EVERY,
    unsafe: bool = False,
    only: tuple[str, ...] = (),
) -> None:
    """Print the diff one pass of the selected rules' fixes would write in ROOT, unproven.

    Each finding's edits are taken unless they collide with ones already taken, as the fixer's
    first pass takes them; nothing is written and no proof runs, so a fix the fixer would withhold
    shows here too.

    Args:
        root: the project whose fixes are planned.
        unsafe: include fixes marked for review, not only safe ones.
        only: print only these files (paths relative to ROOT).
    """
    root = root.resolve()
    with selection.lint(root) as lint:
        repairs = lint.repairs(lint.facts(root), FIXABLE["review" if unsafe else "safe"])
    plan = Plan(root)
    taken = sum(plan.add(repair) for group in repairs.values() for repair in group)
    for name, data in sorted(plan.written().items()):
        if only and name not in only:
            continue
        before = plan.source(name).data.decode().splitlines(keepends=True)
        after = data.decode().splitlines(keepends=True)
        logger.result("".join(difflib.unified_diff(before, after, f"a/{name}", f"b/{name}")))
    logger.result(f"{taken} fixes planned, {plan.deferred} deferred on a collision")
