"""Who holds each card right now: utilization, memory, and the processes on it, by name.

The one screen that says whether a card can take an acquisition, which `facts` does not answer
(hardware, not state) and `jobs` answers only for what this workspace dispatched: a resident
server or another user's run holds a card without ever appearing in a dispatch record.
"""

from __future__ import annotations

import platform
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import psutil
from patos import FrozenModel, FrozenOpenModel

from .machine import Machine

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .units.gpu import GPU

_SCHEMA_VERSION = 1


class Holder(FrozenModel):
    """One process holding a card.

    user: the account running it, empty when the host will not say.
    age_s: how long it has run, seconds.
    command: its command line cut to 80 characters, empty when unreadable.
    used_bytes: the card memory it holds when the driver attributes it.
    """

    pid: int
    user: str = ""
    age_s: int = 0
    command: str = ""
    used_bytes: int = 0


class CardOccupancy(FrozenModel):
    """One card's state at the moment of the reading.

    utilization_pct: the compute utilization the driver reports.
    holders: the compute processes on it, empty when none or when the sensor cannot say.
    """

    index: int
    name: str
    utilization_pct: int = 0
    memory_used_bytes: int = 0
    memory_total_bytes: int = 0
    holders: tuple[Holder, ...] = ()

    @property
    def free(self) -> bool:
        """Whether no process holds the card and it reads idle."""
        return not self.holders and self.utilization_pct < 10


class Occupancy(FrozenOpenModel):
    """Every card of one host with who holds it, the JSON another machine reads back.

    schema_version: format revision, bumped when a field's meaning changes.
    at: when the reading was taken, UTC ISO seconds.
    """

    schema_version: int = _SCHEMA_VERSION
    hostname: str = ""
    at: str = ""
    cards: tuple[CardOccupancy, ...] = ()

    @classmethod
    def collected(cls, gpus: Sequence[GPU] | None = None) -> Occupancy:
        """Read every card of this machine, the probed ones when none are handed in."""
        units = Machine().gpus if gpus is None else gpus
        return cls(
            hostname=platform.node(),
            at=datetime.now(UTC).isoformat(timespec="seconds"),
            cards=tuple(read(unit) for unit in units),
        )


def read(unit: GPU) -> CardOccupancy:
    """One card's occupancy from its live sensors."""
    snapshot = unit.snapshot()
    memory = unit.memory
    return CardOccupancy(
        index=unit.index,
        name=unit.label,
        utilization_pct=snapshot.utilization.gpu_pct,
        memory_used_bytes=memory.used_bytes,
        memory_total_bytes=memory.total_bytes,
        holders=tuple(holder(process.pid, process.used_bytes) for process in snapshot.processes),
    )


def holder(pid: int, used_bytes: int) -> Holder:
    """What the host knows about `pid`, the bare id when the process is gone or private."""
    try:
        process = psutil.Process(pid)
        with process.oneshot():
            return Holder(
                pid=pid,
                user=process.username(),
                age_s=int(datetime.now(UTC).timestamp() - process.create_time()),
                command=" ".join(process.cmdline())[:80],
                used_bytes=used_bytes,
            )
    except psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess:
        return Holder(pid=pid, used_bytes=used_bytes)


def rows(host: str, occupancy: Occupancy) -> list[dict[str, str | int | float | bool]]:
    """The occupancy as one row per card, for tables."""
    return [
        {
            "host": host,
            "card": f"{card.index}: {card.name}",
            "util_pct": card.utilization_pct,
            "memory_gb": round(card.memory_used_bytes / 1e9, 1),
            "of_gb": round(card.memory_total_bytes / 1e9, 1),
            "free": card.free,
            "holders": "; ".join(
                f"{h.pid} {h.user or '?'} {h.age_s // 3600}h {h.command}".strip()
                for h in card.holders
            ),
        }
        for card in occupancy.cards
    ]
