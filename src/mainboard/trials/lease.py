# The card lease: a named refusal between sessions, the shape `Stage` gives claims.
#
# A consumer's `clean_card` checked `gpu_processes()` once at session open, so two sessions started
# moments apart both read a free card and measured beside each other. A lease is a pid and a
# timestamp in a file, not an OS lock: an `flock` releases when its holder dies but has no ttl, so
# a live session that ran too long would hold it forever. The file's own text answers both
# staleness questions, whether the pid lives and how long it has held, and a stale lease is
# reclaimed by unlinking and retrying.

import os
import re
import socket
import time
from pathlib import Path

import psutil

# How long a lease may stand with a live pid behind it: the longest GPU campaign, and no more.
DEFAULT_TTL_S = 24 * 3600.0

# The lease file's prefix, a dotfile so it never mixes into a node listing. The full name carries
# the host and `CUDA_VISIBLE_DEVICES`, since a cluster root is a shared filesystem: on 2026-08-31
# nine PBS jobs on nine GH200 nodes shared one `/work` root, and one refused to start because
# another node's pid happened to name a live process on its own node too.
FILENAME = ".card.lock"


def filename() -> str:
    """The lease file for this host and its visible devices, `.card.lock.<host>[.<devices>]`."""
    host = re.sub(r"[^A-Za-z0-9_.-]", "_", socket.gethostname())
    devices = re.sub(r"[^A-Za-z0-9_-]", "-", os.environ.get("CUDA_VISIBLE_DEVICES", ""))
    return f"{FILENAME}.{host}" + (f".{devices}" if devices else "")


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
    """One process's exclusive hold on a universe's card, released at the end of its session."""

    def __init__(self, path: Path) -> None:
        self.path = path

    @classmethod
    def acquire(cls, root: Path, *, ttl: float = DEFAULT_TTL_S) -> CardLease:
        """Take the lease under `root`, reclaiming a stale one and refusing a live one.

        Exclusive creation closes the race: two sessions opening in the same instant contend on
        one `os.open` that can only succeed once.

        ttl: how long a lease may stand before a live pid behind it no longer excuses it.
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
            if age <= ttl and psutil.pid_exists(pid):
                raise Busy(path, pid, age)
        path.unlink(missing_ok=True)
        lease._write()
        return lease

    def release(self) -> None:
        """Give the lease back, tolerant of it already being gone."""
        self.path.unlink(missing_ok=True)

    def _holder(self) -> tuple[int, float] | None:
        """The pid and opening time the lease file names, None where it cannot be read."""
        try:
            pid, opened = self.path.read_text(encoding="utf-8").split()
            return int(pid), float(opened)
        except OSError, ValueError:
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
