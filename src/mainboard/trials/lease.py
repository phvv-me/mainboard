# THE CARD LEASE: A NAMED REFUSAL BETWEEN SESSIONS, THE SAME SHAPE `Stage` ALREADY GIVES CLAIMS.
#
# `clean_card` in a consumer's conftest checked `gpu_processes()` once at session open and asked
# nothing else, so two sessions started moments apart both read a free card and both proceeded to
# measure beside each other, a race a card-hungry campaign hits by hand. `Stage` already solves
# the neighbouring problem, acquire, check, release, with a refusal naming the specifics, so this
# is that shape spent on a resource that outlives one process's claim rather than one claim's own
# holdings.
#
# A LEASE IS A PID AND A TIMESTAMP IN A FILE, NOT AN OS-LEVEL LOCK. An `flock` releases itself the
# instant its holder dies, which sounds like exactly what staleness needs, but it gives no ttl: a
# session that is very much alive and has simply run for longer than a card measurement ever
# should would hold it forever, and nothing short of killing that process could reclaim it. Naming
# the holder in the file's own text answers both questions from the one place, whether its pid
# still exists and how long it has had the lease, and lets a stale lease (either answer says so)
# be reclaimed by unlinking and retrying rather than by asking a human to find and kill a pid.

import os
import re
import socket
import time
from pathlib import Path

import psutil

# How long a lease may be held before it counts as stale even with a live pid behind it, generous
# enough for the longest GPU campaign this workspace runs and no more.
DEFAULT_TTL_S = 24 * 3600.0

# The one file a universe's root carries for this, a dotfile so it never mixes into a node listing
# and never wants a directory of its own. IT IS NAMED FOR THE MACHINE, because a universe root on
# a cluster is a shared filesystem: on 2026-08-31 nine PBS jobs on nine GH200 nodes shared one
# `/work` root, and a job refused to start because the pid another node had written happened to
# name a live process on its own node too. A card is a property of one host, and of one device
# set when `CUDA_VISIBLE_DEVICES` narrows it, so the lease carries both in its name and two hosts
# never read each other's.
FILENAME = ".card.lock"


def filename() -> str:
    """The lease file for this host and its visible devices, `.card.lock.<host>[.<devices>]`."""
    host = re.sub(r"[^A-Za-z0-9_.-]", "_", socket.gethostname())
    devices = re.sub(r"[^A-Za-z0-9_-]", "-", os.environ.get("CUDA_VISIBLE_DEVICES", ""))
    return f"{FILENAME}.{host}" + (f".{devices}" if devices else "")


class Busy(RuntimeError):
    """Another process holds the card lease, live and inside its ttl.

    path: the lease file. pid: the process holding it. age: how long it has held it, in seconds.
    """

    def __init__(self, path: Path, pid: int, age: float) -> None:
        self.pid = pid
        self.age = age
        super().__init__(
            f"{path} is held by pid {pid} for {age:.0f}s; refusing to measure beside it. If that "
            f"process is gone, delete {path} and retry."
        )


class CardLease:
    """One process's exclusive hold on a universe's card, released at the end of its session.

    path: where the lease is recorded.
    """

    def __init__(self, path: Path) -> None:
        self.path = path

    @classmethod
    def acquire(cls, root: Path, *, ttl: float = DEFAULT_TTL_S) -> CardLease:
        """Take the lease under `root`, reclaiming a stale one and refusing a live one.

        Exclusive creation is what closes the race `clean_card` never noticed: two sessions
        opening within the same instant contend on the one `os.open` that can only succeed once,
        rather than both reading a free card before either had written anything.

        root: the universe root the lease is scoped to. ttl: how long a lease may stand before a
            live pid behind it no longer excuses it.
        """
        path = root / filename()
        path.parent.mkdir(parents=True, exist_ok=True)
        lease = cls(path)
        if lease._write():
            return lease
        holder = lease._holder()
        if holder is not None:
            pid, opened = holder
            age = time.time() - opened
            if age <= ttl and _alive(pid):
                raise Busy(path, pid, age)
        path.unlink(missing_ok=True)
        lease._write()
        return lease

    def release(self) -> None:
        """Give the lease back, tolerant of it already being gone."""
        self.path.unlink(missing_ok=True)

    def _holder(self) -> tuple[int, float] | None:
        """The pid and opening time the lease file names, None where it cannot be read at all."""
        try:
            pid, opened = self.path.read_text(encoding="utf-8").split()
        except OSError, ValueError:
            return None
        try:
            return int(pid), float(opened)
        except ValueError:
            return None

    def _write(self) -> bool:
        """Create the lease file naming this process, refusing rather than overwriting one."""
        try:
            handle = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return False
        with os.fdopen(handle, "w", encoding="utf-8") as opened:
            opened.write(f"{os.getpid()} {time.time()}")
        return True


def _alive(pid: int) -> bool:
    """Whether `pid` still names a running process on this machine."""
    return psutil.pid_exists(pid)
