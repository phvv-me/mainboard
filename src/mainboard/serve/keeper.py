"""The keeper `mb serve`'s unit runs: one supervisor at a time, and nothing of it left behind.

The unit runs it by path with the tool's own interpreter, `python -I keeper.py STATE -- COMMAND`,
and it imports the standard library and psutil alone, so the resident process costs a bare
interpreter (22 MB) rather than the whole tool (150 MB).

process-compose starts every child in a process group of its own and macOS has no subreaper, so a
supervisor that dies orphans every child, and killing its group reaches none of them (spike,
2026-10-28). Its children stay in its session, though: the keeper starts it as a session leader,
and a process group never leaves its session. So the ledger names the supervisor once, by pid and
start time, and whoever stops it (the keeper itself, or the next keeper after a crash) kills every
process still in that session, grandchildren whose parent died included. No child can escape that
way by being restarted between two looks, which a ledger of process groups allowed.

The lock makes it the only keeper of its directory. A second one exits 75 at once, and the
service manager starts the unit again a few seconds later.
"""

import fcntl
import json
import math
import os
import signal
import stat
import subprocess
import sys
import time
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import psutil

if TYPE_CHECKING:
    from collections.abc import Sequence
    from types import FrameType

# How often the socket's mode is looked at while the supervisor runs.
_TICK = 1.0
# How long a swept process has between SIGTERM and SIGKILL.
_GRACE = 1.0
# How long a stopping supervisor may take before it is killed: it stops its children one after
# another, each within its own timeout, and a hung one must not keep the keeper for good.
_ABANDON = 120.0
# The socket's mode: its API starts and stops processes, so only its owner may connect.
_SOCKET_MODE = 0o600
# `EX_TEMPFAIL`: another keeper holds the lock, and the service manager tries again later.
_HELD = 75
# `EX_USAGE`: the command line is not `STATE -- COMMAND...`.
_USAGE = 64
# process-compose never rotates its own log; past this size a start moves it aside once.
_ROTATED = 8 << 20
# The signals that stop the keeper. A launchd bootout reaches it as its terminal's hangup.
_STOPS = (signal.SIGHUP, signal.SIGINT, signal.SIGTERM)


class Layout(NamedTuple):
    """Where one served workspace keeps its supervisor's files, all in one 0700 directory.

    state: that directory, `.mb/serve` at the workspace root.
    """

    state: Path

    @property
    def compose(self) -> Path:
        """The generated process-compose YAML."""
        return self.state / "compose.yaml"

    @property
    def ledger(self) -> Path:
        return self.state / "ledger.json"

    @property
    def lock(self) -> Path:
        return self.state / "keeper.lock"

    @property
    def log(self) -> Path:
        """What the keeper says, and what process-compose writes to standard error."""
        return self.state / "keeper.log"

    @property
    def logs(self) -> Path:
        """The children's own logs, one rotated file each."""
        return self.state / "logs"

    @property
    def socket(self) -> Path:
        """process-compose's API, a unix socket and never a port."""
        return self.state / "pc.sock"

    @property
    def supervisor_log(self) -> Path:
        """process-compose's own log, which it does not rotate."""
        return self.state / "pc.log"

    @property
    def unit_log(self) -> Path:
        """What the service manager's unit says before the keeper opens its own log."""
        return self.state / "unit.log"

    def guarded(self) -> bool:
        """Whether the socket exists at `_SOCKET_MODE`, the only mode it may have."""
        try:
            return stat.S_IMODE(self.socket.stat().st_mode) == _SOCKET_MODE
        except FileNotFoundError:
            return False


class Leader(NamedTuple):
    """A process as the ledger names it, the leader of the session it started.

    pid: its process id, which is also its session's id.
    started: its start time as psutil reads it, which tells a reused pid apart.
    """

    pid: int
    started: float

    @classmethod
    def of(cls, pid: int) -> Leader | None:
        """The process `pid` is now, None when no such process exists."""
        started = Leader._started(pid)
        return cls(pid=pid, started=started) if started < math.inf else None

    @staticmethod
    def _session(pid: int) -> int:
        """Process `pid`'s session id, 0 for one gone or closed to this user."""
        try:
            return os.getsid(pid)
        except OSError:
            return 0

    @staticmethod
    def _started(pid: int) -> float:
        """Process `pid`'s start time, infinity for one gone or closed to this user."""
        try:
            return psutil.Process(pid).create_time()
        except psutil.Error:
            return math.inf

    def alive(self) -> bool:
        """Whether this very process still runs."""
        return Leader.of(self.pid) == self

    def members(self) -> list[psutil.Process]:
        """Every process still in this leader's session, itself included while it runs.

        A dead leader's pid may already lead another session: a member then counts only when
        it started before that newcomer did, which every process of the old session did.
        """
        now = Leader.of(self.pid)
        until = now.started if now is not None and now != self else math.inf
        return [
            process
            for process in psutil.process_iter()
            if Leader._session(process.pid) == self.pid
            and self.started <= Leader._started(process.pid) < until
        ]


class Ledger:
    """The keeper and the supervisor it started, on disk so a crash leaves their names behind.

    path: the JSON file, written whole and replaced, never edited in place.
    """

    @staticmethod
    def _decoded(text: str) -> tuple[Leader, Leader]:
        """The keeper and the supervisor a ledger's text names, raising on a torn one."""
        written = json.loads(text)
        return Leader(*written["keeper"]), Leader(*written["supervisor"])

    def __init__(self, path: Path) -> None:
        self.path = path

    def _read(self) -> tuple[Leader | None, Leader | None]:
        """The keeper and the supervisor recorded; neither when the ledger is missing or torn."""
        try:
            return Ledger._decoded(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError, ValueError, KeyError, TypeError:
            return None, None

    def keeper(self) -> Leader | None:
        """The keeper that wrote the ledger, None when there is none."""
        return self._read()[0]

    def members(self) -> list[psutil.Process]:
        """Every process still in the recorded supervisor's session."""
        return supervisor.members() if (supervisor := self.supervisor()) is not None else []

    def record(self, *, keeper: Leader, supervisor: Leader) -> None:
        """Replace the ledger in one rename, so a reader never finds half of it."""
        partial = self.path.with_suffix(".partial")
        written = {"keeper": list(keeper), "supervisor": list(supervisor)}
        partial.write_text(json.dumps(written), encoding="utf-8")
        partial.replace(self.path)

    def supervisor(self) -> Leader | None:
        """The supervisor the ledger names, None when there is none."""
        return self._read()[1]

    def sweep(self) -> int:
        """Kill every process still in the recorded session, TERM and then KILL.

        Answers how many processes were still there to kill. The ledger is forgotten once its
        session is empty and kept otherwise, so the next sweep tries again.
        """
        members = self.members()
        for process in members:
            with suppress(psutil.Error):
                process.terminate()
        _, alive = psutil.wait_procs(members, timeout=_GRACE)
        for process in alive:
            with suppress(psutil.Error):
                process.kill()
        psutil.wait_procs(alive, timeout=_GRACE)
        if not self.members():
            self.path.unlink(missing_ok=True)
        return len(members)


class Keeper:
    """Runs one supervisor command under the lock, the ledger and the signals that stop it.

    layout: the state directory's files.
    command: the supervisor's argv, its environment already cut down to the allowlist.
    """

    def __init__(self, layout: Layout, command: Sequence[str]) -> None:
        self.layout = layout
        self.command = command
        self.ledger = Ledger(layout.ledger)
        # When a stop was asked for, on the monotonic clock; zero while none was.
        self.asked = 0.0
        self.supervisor: subprocess.Popen[bytes] | None = None

    def _guard(self) -> None:
        """Hold the socket at 0600: process-compose binds it at 0700 under umask 077."""
        if not self.layout.guarded():
            with suppress(FileNotFoundError):
                self.layout.socket.chmod(_SOCKET_MODE)

    def _kept(self) -> int:
        """Sweep what a crash left, run the supervisor until it exits, then sweep again.

        Answers the supervisor's exit status, 128 plus the signal that ended it, as a shell does.
        The stop handlers come first, so a stop asked for while it starts still reaches it.
        """
        for number in _STOPS:
            signal.signal(number, self._stop)
        if swept := self.ledger.sweep():
            self._say(f"swept {swept} processes an earlier run left behind")
        supervisor = self._started()
        self._watched(supervisor)
        swept = self.ledger.sweep()
        code = supervisor.returncode
        self._say(f"the supervisor exited {code}; swept {swept} processes it left")
        return code if code >= 0 else 128 - code

    def _say(self, line: str) -> None:
        """Append one timestamped line to the keeper's log, its only output."""
        with self.layout.log.open("a", encoding="utf-8") as log:
            log.write(f"{datetime.now(UTC).isoformat(timespec='seconds')} {line}\n")

    def _started(self) -> subprocess.Popen[bytes]:
        """Start the supervisor as a session leader and record it before anything else runs.

        Its own log is moved aside first once it passes `_ROTATED` bytes, and a stop asked for
        before the ledger named it is passed on at once.
        """
        self.layout.socket.unlink(missing_ok=True)
        with suppress(FileNotFoundError):
            if self.layout.supervisor_log.stat().st_size > _ROTATED:
                self.layout.supervisor_log.replace(self.layout.state / "pc.log.1")
        with self.layout.log.open("a", encoding="utf-8") as log:
            started = self.supervisor = subprocess.Popen(
                self.command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=log,
                cwd=self.layout.state,
                start_new_session=True,
            )
        self.ledger.record(
            keeper=Leader(pid=os.getpid(), started=psutil.Process().create_time()),
            supervisor=Leader.of(started.pid) or Leader(pid=started.pid, started=time.time()),
        )
        self._say(f"started the supervisor, pid {started.pid}")
        if self.asked:
            started.send_signal(signal.SIGTERM)
        return started

    def _stop(self, number: int, frame: FrameType | None) -> None:
        """Ask the supervisor to stop its children; the loop then waits for it and sweeps."""
        del frame
        self._say(f"stopping on {signal.Signals(number).name}")
        self.asked = self.asked or time.monotonic()
        if self.supervisor is not None:
            with suppress(ProcessLookupError):
                self.supervisor.send_signal(signal.SIGTERM)

    def _watched(self, supervisor: subprocess.Popen[bytes]) -> None:
        """Guard the socket every tick until `supervisor` exits.

        A supervisor still running `_ABANDON` seconds after a stop was asked for is killed.
        """
        while supervisor.poll() is None:
            self._guard()
            if self.asked and time.monotonic() - self.asked > _ABANDON:
                self._say(f"the supervisor did not stop within {_ABANDON:.0f}s, so it is killed")
                with suppress(ProcessLookupError):
                    os.killpg(supervisor.pid, signal.SIGKILL)
            time.sleep(_TICK)

    def run(self) -> int:
        """Keep the supervisor, or exit 75 at once when another keeper holds the lock."""
        os.umask(0o077)
        with self.layout.lock.open("a", encoding="utf-8") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                self._say(f"another keeper holds {self.layout.lock}, so this one exits")
                return _HELD
            return self._kept()


def main(argv: Sequence[str]) -> int:
    """`keeper.py STATE -- COMMAND...`, answering the exit status."""
    match argv:
        case [state, "--", *command] if command:
            return Keeper(Layout(Path(state)), command).run()
        case _:
            sys.stderr.write("usage: keeper.py STATE -- COMMAND...\n")
            return _USAGE


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
