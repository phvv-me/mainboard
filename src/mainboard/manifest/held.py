# The machines this workspace is holding right now, and the host profile each one answers to.
#
# A held rental is an ssh host for as long as it lives, so every verb that takes `--on` has to
# resolve its alias the way it resolves `gold`. The manifest cannot say so, since the machine did
# not exist when the manifest was written and will not exist next week, so the profile lives here
# beside the dispatch state and the loader lays it over the declared hosts. The file is JSON
# because a program writes it and a program reads it back.

import os
from pathlib import Path

from patos import FrozenModel
from pydantic import AwareDatetime, TypeAdapter

from ..core.project import Project
from .schema.host import HostProfile


class Held(FrozenModel):
    """One rented machine kept for a session, reachable as an ordinary ssh host.

    alias: the ssh alias and host name every verb reaches it by.
    provider: the provider host it was rented through, `vast` say.
    handle: the provider's own id for the rental, which ends it.
    gpu: the card it was rented with, as the provider spells it.
    usd_hr: the hourly rate its quote carried, None when the provider quoted none.
    deadline: when it is released whether or not anyone asks.
    profile: the ssh host profile it answers to, its provider's sync scope and variables kept.
    """

    alias: str
    provider: str
    handle: str
    gpu: str = ""
    usd_hr: float | None = None
    deadline: AwareDatetime
    profile: HostProfile


# How the file spells the holdings: one JSON list of records.
_LISTED = TypeAdapter(list[Held])


class Holdings:
    """The file of held machines inside one workspace's dispatch state.

    root: the workspace root.
    """

    def __init__(self, root: Path) -> None:
        self.path = root / Project().out_dir / "dispatch" / "holds.json"

    def read(self) -> dict[str, Held]:
        """Every held machine by alias, none when nothing was ever held here."""
        try:
            text = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return {}
        return {held.alias: held for held in _LISTED.validate_json(text)}

    def profiles(self) -> dict[str, HostProfile]:
        """The host profile of every held machine by alias, what the loader lays over `[hosts]`."""
        return {alias: held.profile for alias, held in self.read().items()}

    def save(self, held: Held) -> None:
        """Record `held`, replacing whatever was recorded under its alias."""
        self._write({**self.read(), held.alias: held})

    def drop(self, alias: str) -> None:
        """Forget `alias`, a no-op for one that was never held."""
        self._write({name: held for name, held in self.read().items() if name != alias})

    def _write(self, holdings: dict[str, Held]) -> None:
        """Replace the file in one rename, so a reader never sees half of it."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        pending = self.path.with_suffix(".part")
        pending.write_bytes(_LISTED.dump_json(list(holdings.values()), indent=2))
        os.replace(pending, self.path)
