"""Asks: acts a policy check or a handoff trigger could not allow, waiting for the owner's answer.

An ask is a proposal and carries no authority, so it travels as telemetry: an OTLP line in mb's
spool, which the collect pass delivers into the telemetry lake, and it is read back from both. Who
files an ask never answers it: answers live with the grants, which only the owner writes. One act
on one target is one ask whatever asks it, its id minted from the two, so a pass asking every five
minutes files one line until the owner answers; asked again after an answer, it is pending anew.
"""

import os
import unicodedata
from datetime import UTC, datetime
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Self
from uuid import uuid5

from patos import FrozenModel
from pydantic import AfterValidator, AwareDatetime, ValidationError

from ..constants import ASK_ACT, ASK_EVENT, ASK_TARGET
from ..core.errors import MissionError
from ..core.events import Event, EventSpool
from ..enums import Act
from ..log import logger
from ..state.telemetry import TelemetryLake
from .constants import ASK_NAMESPACE
from .grants import Target, said

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    from ..core.events.identity import Attribute
    from .grants import Grants

_LAKE_SQL = "SELECT asked, why, act, target FROM asks"

# One filed line as both sources give it: when, its body, its attributes.
type Line = tuple[datetime, str, Mapping[str, Attribute]]


def _printable(text: str) -> str:
    """`text` as an ask's why, refused if it holds a control, format or unassigned character.

    Anyone holding the sink's token can file a why, and a control in it could redraw what the
    owner reads before answering.
    """
    if held := "".join(sorted({c for c in text if unicodedata.category(c).startswith("C")})):
        raise ValueError(f"the why holds the characters {held!r}, which could redraw a terminal")
    return text


@cache
def _skipped(path: Path, fault: str) -> None:
    """Say once per process that `path` was skipped for `fault`; every pass reads it again."""
    logger.warning(f"skipped {path.name}: {fault}")


class Ask(FrozenModel):
    """One act on one target that waits for the owner, and why it could not go ahead.

    at: when it was filed; an act asked again files a later one under the same id.
    """

    act: Act
    target: Target
    why: Annotated[str, AfterValidator(_printable)]
    at: AwareDatetime

    @classmethod
    def of(cls, line: Line) -> Self:
        """The ask one filed line carries; `ValidationError` for a line that is no ask."""
        at, body, attributes = line
        fields = {"act": attributes.get(ASK_ACT), "target": attributes.get(ASK_TARGET)}
        return cls.model_validate({**fields, "why": body, "at": at})

    @property
    def id(self) -> str:
        """The same for every ask of this act on this target, so asking twice is asking once."""
        return str(uuid5(ASK_NAMESPACE, f"{self.act} {self.target}"))

    def event(self) -> Event:
        """This ask as the event the spool carries."""
        attributes: dict[str, Attribute] = {ASK_ACT: str(self.act), ASK_TARGET: self.target}
        return Event(name=ASK_EVENT, at=self.at, body=self.why, attributes=attributes)


class Asks:
    """One workspace's asks, in its spool and its telemetry lake.

    root: the workspace whose `.mb/telemetry/` holds both.
    """

    def __init__(self, root: Path) -> None:
        self.lake = TelemetryLake(root=root)

    def _delivered(self) -> list[tuple[datetime, str | None, str | None, str | None]]:
        """The rows of the lake's `asks` view; refused while the sink has not made the lake."""
        with self.lake.read() as connection:
            return connection.execute(_LAKE_SQL).fetchall()

    def _filed(self) -> dict[str, Ask]:
        """The latest ask of each id.

        A line that is no ask is skipped and said, since anyone holding the sink's token can
        send one.
        """
        latest: dict[str, Ask] = {}
        for line in self._lines():
            try:
                ask = Ask.of(line)
            except ValidationError as fault:
                logger.warning(f"skipped an ask line that is no ask: {said(fault)}")
                continue
            if ask.id not in latest or ask.at > latest[ask.id].at:
                latest[ask.id] = ask
        return latest

    def _lines(self) -> Iterator[Line]:
        """Every ask line still in the spool, then every one the lake holds (naive UTC there).

        A spool file holding a line that is no event is skipped whole and said once.
        """
        for path in sorted(self.lake.spool.glob("policy-*.jsonl")):
            try:
                events = EventSpool.read(path).events
            except ValueError as fault:
                _skipped(path, str(fault))
                continue
            yield from ((e.at, e.body, e.attributes) for e in events if e.name == ASK_EVENT)
        try:
            rows = self._delivered()
        except MissionError:
            return
        for at, why, act, target in rows:
            named = {ASK_ACT: act, ASK_TARGET: target}
            attributes = {key: value for key, value in named.items() if value is not None}
            yield at.replace(tzinfo=UTC), why or "", attributes

    def file(self, ask: Ask, grants: Grants) -> Ask:
        """Spool `ask` unless the same ask already waits, and answer the one that waits."""
        waiting = {known.id: known for known in self.pending(grants, at=ask.at)}
        if ask.id in waiting:
            return waiting[ask.id]
        with EventSpool(self.lake.spool / f"policy-{os.getpid()}.jsonl") as spool:
            spool.append(ask.event())
        return ask

    def pending(self, grants: Grants, *, at: datetime) -> list[Ask]:
        """The asks the owner has not answered since they were filed, and no live grant covers."""
        waiting = [
            ask
            for ask in self._filed().values()
            if ((answered := grants.answered(ask.id)) is None or ask.at > answered)
            and grants.covering(ask.act, ask.target, at=at) is None
        ]
        return sorted(waiting, key=lambda ask: ask.at)
