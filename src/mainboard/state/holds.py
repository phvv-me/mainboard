# The lake's record of held machines: `holds_log`, whose `holds` view is the current hold per
# alias. A module of its own so a manifest reaches it lazily, never loading the database engine in
# a workspace that holds nothing.

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from sqlalchemy import select

from . import schema
from .lake import Lake

if TYPE_CHECKING:
    from pathlib import Path


class HoldLog:
    """The held-machine log of the workspace at `root`."""

    def __init__(self, root: Path) -> None:
        self.session = Lake.at(root).session()

    def records(self) -> list[str]:
        """Every current hold's JSON record, by alias."""
        holds = schema.holds
        rows = self.session.rows(select(holds.c.held).order_by(holds.c.alias))
        return [held for (held,) in rows]

    def append(self, alias: str, **fields: str | bool) -> None:
        """Record `fields` as `alias`'s hold, `dropped=True` forgetting it."""
        record = {"ts": datetime.now(UTC), "alias": alias, **fields}
        self.session.append(schema.holds_log, [record])
