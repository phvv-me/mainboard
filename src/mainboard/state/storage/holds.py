# The machines this workspace is holding and their host profiles, kept in the lake's `holds_log`,
# whose `holds` view is the current hold per alias. A held rental resolves under `--on` like
# `gold` but outlives no manifest, so `holding` lays the held profiles over the declared hosts. A
# workspace with no lake yet holds nothing, and reading its holds never creates one.

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from patos import FrozenModel
from pydantic import AwareDatetime
from sqlalchemy import select

from ...manifest.loading import load
from ...manifest.schema.compute.host import HostProfile
from .. import schema
from ..lake import Lake

if TYPE_CHECKING:
    from pathlib import Path

    from ...manifest import Manifest


class HoldLog:
    """The held-machine log of the workspace at `root`."""

    def __init__(self, root: Path) -> None:
        self.session = Lake.at(root).session()

    def append(self, alias: str, **fields: str | bool) -> None:
        """Record `fields` as `alias`'s hold, `dropped=True` forgetting it."""
        record = {"ts": datetime.now(UTC), "alias": alias, **fields}
        self.session.append(schema.holds_log, [record])

    def records(self) -> list[str]:
        """Every current hold's JSON record, by alias."""
        holds = schema.holds
        rows = self.session.rows(select(holds.c.held).order_by(holds.c.alias))
        return [held for (held,) in rows]


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

    def __init__(self, root: Path, *, home: Path) -> None:
        """home: where the workspace's lake lives, `Header.lake_home`."""
        self.root, self.home = root, home

    def _append(self, alias: str, **fields: str | bool) -> None:
        self._log().append(alias, **fields)

    def _log(self) -> HoldLog:
        """The workspace lake's hold log."""
        return HoldLog(self.root)

    def drop(self, alias: str) -> None:
        """Forget `alias`, if held."""
        self._append(alias, dropped=True)

    def profiles(self) -> dict[str, HostProfile]:
        """The host profile of every held machine by alias."""
        return {alias: held.profile for alias, held in self.read().items()}

    def read(self) -> dict[str, Held]:
        """Every held machine by alias, none in a workspace that has no lake yet."""
        if not (self.home / "lake.sqlite").is_file():
            return {}
        # A hold an older release recorded may carry a profile field this release dropped.
        held = (Held.model_validate_json(row, extra="ignore") for row in self._log().records())
        return {machine.alias: machine for machine in held}

    def save(self, held: Held) -> None:
        """Record `held`, replacing whatever was recorded under its alias."""
        self._append(held.alias, held=held.model_dump_json())


def holding(path: Path) -> Manifest:
    """The manifest at `path` with its workspace's held machines standing behind `[hosts]`.

    They are read from the lake only when a lookup reaches them, so a held alias resolves like a
    declared one and a declared one never waits on the lake.
    """
    manifest = load(path)
    home = manifest.workspace.lake_home(path.parent)
    return manifest.holding(Holdings(path.parent, home=home).profiles)
