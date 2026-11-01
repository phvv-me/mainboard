from pydantic import Field

from ....core.base import Declared


class Budget(Declared):
    """What rentals may cost over a trailing window, a second line under each provider's own cap.

    usd: the most the lake's `costs` may show billed over the window, a new rental's budget
        included.
    days: how far back the window reaches.
    """

    usd: float = Field(gt=0, allow_inf_nan=False)
    days: int = Field(default=7, gt=0)

    def allows(self, *, spent_usd: float, max_usd: float) -> bool:
        """Whether a rental budgeted at `max_usd` still fits beside `spent_usd` billed already."""
        return spent_usd + max_usd <= self.usd


class Policy(Declared):
    """The `[policy]` table: limits on what runs, pushes and spends with no one at a terminal.

    It only ever limits, since an agent can edit this file. What lifts a limit is a grant, which
    only the owner writes (`serve grant`, `serve answer`), into a file outside the workspace.

    rent: the spend line every rental is checked against beside its provider's cap; none
        declared, the providers' caps alone bound what rentals cost.
    """

    rent: Budget | None = None
