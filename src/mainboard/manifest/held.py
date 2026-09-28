# The machines this workspace is holding and their host profiles. A held rental resolves under
# `--on` like `gold`, but outlives no manifest, so its profile lives in the workspace lake's
# `holds_log` and the loader lays it over the declared hosts. A workspace with no lake yet holds
# nothing, and reading its holds never creates one.

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from patos import FrozenModel
from pydantic import AwareDatetime

from ..core.project import Project
from .schema.host import HostProfile

if TYPE_CHECKING:
    from pathlib import Path

    from ..state.lake import Session


class Held(FrozenModel):
    """One rented machine kept for a session, reachable as an ordinary ssh host.

    provider: the provider host it was rented through, `vast` say.
    handle: the provider's own id for the rental, which ends it.
    gpu: the card, as the provider spells it.
    usd_hr: the quoted hourly rate, None when none was quoted.
    deadline: when it is released whether or not anyone asks.
    profile: its ssh profile, keeping its provider's sync scope and variables.
    """

    alias: str
    provider: str
    handle: str
    gpu: str = ""
    usd_hr: float | None = None
    deadline: AwareDatetime
    profile: HostProfile


class Holdings:
    """The held machines of one workspace, the lake's `holds` view."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def read(self) -> dict[str, Held]:
        """Every held machine by alias, none in a workspace that has no lake yet."""
        if not (Project().out(self.root) / "lake.sqlite").is_file():
            return {}
        rows = self._session().rows("SELECT held FROM lake.holds ORDER BY alias")
        return {held.alias: held for held in (Held.model_validate_json(row) for (row,) in rows)}

    def profiles(self) -> dict[str, HostProfile]:
        """The host profile of every held machine by alias."""
        return {alias: held.profile for alias, held in self.read().items()}

    def save(self, held: Held) -> None:
        """Record `held`, replacing whatever was recorded under its alias."""
        self._append(held.alias, held=held.model_dump_json())

    def drop(self, alias: str) -> None:
        """Forget `alias`, if held."""
        self._append(alias, dropped=True)

    def _append(self, alias: str, **fields: object) -> None:
        self._session().append("holds_log", [{"ts": datetime.now(UTC), "alias": alias, **fields}])

    def _session(self) -> Session:
        """The workspace lake's shared session, imported here so loading a manifest in a
        workspace holding nothing never loads the database engine."""
        from ..state.lake import Lake

        return Lake.at(self.root).session()
