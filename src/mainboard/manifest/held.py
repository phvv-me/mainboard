# The machines this workspace is holding and their host profiles. A held rental resolves under
# `--on` like `gold`, but outlives no manifest, so its profile lives beside the dispatch state
# (JSON, program to program) and the loader lays it over the declared hosts.

import os
from pathlib import Path

from patos import FrozenModel
from pydantic import AwareDatetime, TypeAdapter

from ..core.project import Project
from .schema.host import HostProfile


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


_LISTED = TypeAdapter(list[Held])


class Holdings:
    """The file of held machines inside one workspace's dispatch state."""

    def __init__(self, root: Path) -> None:
        self.path = root / Project().out_dir / "dispatch" / "holds.json"

    def read(self) -> dict[str, Held]:
        """Every held machine by alias."""
        try:
            text = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return {}
        return {held.alias: held for held in _LISTED.validate_json(text)}

    def profiles(self) -> dict[str, HostProfile]:
        """The host profile of every held machine by alias."""
        return {alias: held.profile for alias, held in self.read().items()}

    def save(self, held: Held) -> None:
        """Record `held`, replacing whatever was recorded under its alias."""
        self._write({**self.read(), held.alias: held})

    def drop(self, alias: str) -> None:
        """Forget `alias`, if held."""
        self._write({name: held for name, held in self.read().items() if name != alias})

    def _write(self, holdings: dict[str, Held]) -> None:
        """Replace the file in one rename, so a reader never sees half of it."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        pending = self.path.with_suffix(".part")
        pending.write_bytes(_LISTED.dump_json(list(holdings.values()), indent=2))
        os.replace(pending, self.path)
