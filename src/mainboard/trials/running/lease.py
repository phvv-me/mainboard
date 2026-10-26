# The card lease: a named refusal between sessions, the shape `Stage` gives claims.
#
# A consumer's `clean_card` checked `gpu_processes()` once at session open, so two sessions started
# moments apart both read a free card and measured beside each other. A lease is a pid, a time and
# a token in a file, not an OS lock: an `flock` releases when its holder dies but has no ttl, so a
# live session that ran too long would hold it forever. The file's own text answers both staleness
# questions, whether the pid lives and how long it has held.
#
# ONE LEASE PER PHYSICAL CARD, HOST-WIDE (owner decision Q05, 2026-10-10). It was a file under each
# universe's root named by `CUDA_VISIBLE_DEVICES`, so two experiments, or the masks `0,1` and `1`,
# never contended for the same card (B12). It is keyed by the card's UUID now, in the user's
# cache, the host's name in the file so a cluster's shared home keeps each node's apart.
#
# EVERY STEP IS ATOMIC (B22). A lease is written whole under a private name and linked into place,
# so no reader sees it empty and takes it for stale; a stale one is renamed aside, which only one
# reclaimer can win; and a release removes the lease only while it is still this one's, so an
# owner whose lease was reclaimed never deletes its successor's.

import os
import re
import socket
import time
import uuid
from contextlib import suppress
from pathlib import Path

import psutil

from ...state.storage.database import cache_home

# How long a lease may stand with a live pid behind it: the longest GPU campaign, and no more.
_DEFAULT_TTL_S = 24 * 3600.0


class Busy(RuntimeError):
    """Another process holds the card lease, live and inside its ttl.

    age: how long it has held the lease, in seconds.
    """

    def __init__(self, path: Path, pid: int, age: float) -> None:
        self.pid = pid
        self.age = age
        super().__init__(
            f"{path} is held by pid {pid} for {age:.0f}s; refusing to measure beside it. If that "
            f"process is gone, delete {path} and retry."
        )


class CardLease:
    """One process's exclusive hold on a card, released at the end of its session."""

    @classmethod
    def acquire(
        cls, card: str, *, home: Path | None = None, ttl: float = _DEFAULT_TTL_S
    ) -> CardLease:
        """Take `card`'s lease, reclaiming a stale one and refusing a live one.

        card: the card's identity, its UUID, which no device mask renumbers.
        home: the folder leases live in, the user's cache when None.
        ttl: how long a lease may stand before a live pid behind it no longer excuses it.
        """
        folder = home or cache_home() / "mb" / "leases"
        folder.mkdir(parents=True, exist_ok=True)
        lease = cls(folder / f"card.{_safe(socket.gethostname())}.{_safe(card)}")
        if lease._placed():
            return lease
        holder = lease._holder()
        if holder is not None:
            pid, opened, _ = holder
            age = time.time() - opened
            if age <= ttl and psutil.pid_exists(pid):
                raise Busy(lease.path, pid, age)
        stale = lease.path.with_name(f"{lease.path.name}.stale.{lease.token}")
        with suppress(FileNotFoundError):
            lease.path.rename(stale)
            stale.unlink(missing_ok=True)
        if not lease._placed():
            # Another session reclaimed it in between and holds it now.
            pid, opened, _ = lease._holder() or (0, time.time(), "")
            raise Busy(lease.path, pid, time.time() - opened)
        return lease

    def __init__(self, path: Path) -> None:
        self.path = path
        self.token = uuid.uuid4().hex

    def _holder(self) -> tuple[int, float, str] | None:
        """The pid, opening time and token the lease file names, None where it cannot be read."""
        with suppress(OSError, ValueError):
            pid, opened, token = self.path.read_text(encoding="utf-8").split()
            return int(pid), float(opened), token
        return None

    def _placed(self) -> bool:
        """Write the lease whole under a private name, then link it into place.

        It answers False when a lease already stands there.
        """
        draft = self.path.with_name(f"{self.path.name}.draft.{self.token}")
        draft.write_text(f"{os.getpid()} {time.time()} {self.token}", encoding="utf-8")
        try:
            os.link(draft, self.path)
        except FileExistsError:
            return False
        finally:
            draft.unlink(missing_ok=True)
        return True

    def release(self) -> None:
        """Give the lease back while it is still this one's; a reclaimed lease is left alone.

        Only a lease past its ttl is reclaimed, so this one changes hands between the read and
        the unlink only when its own live holder outran the ttl at that very moment.
        """
        if (held := self._holder()) is not None and held[2] == self.token:
            self.path.unlink(missing_ok=True)


def _safe(name: str) -> str:
    """`name` as one filename component."""
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name)
