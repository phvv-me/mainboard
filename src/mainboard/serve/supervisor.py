"""`mb serve`: the registry's children under one process-compose, kept in this machine's one unit.

    launchd user agent over `ssh -tt localhost` | systemd --user service
    └─ keeper: `python -I keeper.py`, holding the lock and the ledger, stopping on a signal
       └─ process-compose: a 0600 socket and no port, no dotenv, its log in `.mb/serve/`
          └─ every declared child that is not off

`start` replaces whatever runs, `stop` removes the unit and waits until nothing of the tree is
left, and `status` reads the supervisor's API, the ledger and the processes themselves: names,
pids and states, never a process's arguments and never an environment's values.
"""

import ipaddress
import os
import signal
import sys
import time
from contextlib import suppress
from datetime import timedelta
from inspect import getfile
from typing import TYPE_CHECKING

import psutil

from ..core.constants.units import NANOSECONDS_PER_SECOND
from ..core.environ import Dotenv
from ..core.errors import MissionError
from ..core.execution.tool import ToolRunner
from ..core.project import Project
from ..core.section import Section
from ..enums import Verdict
from . import keeper
from .client import Client, Unanswered
from .compose import Compose
from .constants import SERVICES
from .enums import Kind
from .keeper import Layout, Ledger

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence
    from pathlib import Path

    from ..core.services import Unit
    from .client import Process, ProjectState
    from .compose import Programs
    from .constants import Service

# The supervisor, and the releases whose flags and YAML keys this module writes (`-U`,
# `--disable-dotenv`, `is_strict`, `disable_env_expansion`, `is_template_disabled`).
_SUPERVISOR = "process-compose"
_WANTED = ">=1.124, <2"
# A unix socket's path fits in `sun_path`, 104 bytes on macOS (108 on Linux), its NUL included.
_SOCKET_BYTES = 104
# How long a started unit may take to answer: an ssh login, the keeper, then process-compose.
_STARTING = 30.0
# How long the YAML check may take, and how often a wait looks again.
_VALIDATING = 20.0
_POLL = 0.25
# What a stop waits beyond the children's own timeouts before it sweeps what is left.
_MARGIN = 10.0
# What the API calls a resident child whose stop was asked for, so a stop in flight is no failure.
_STOPPING = frozenset({"Terminating", "Restarting", "Launching", "Pending"})


class Supervisor:
    """`mb serve` for one workspace: start, stop and status of its whole process tree.

    root: the workspace served; its state is `.mb/serve` under it.
    programs: where process-compose and the children's programs are.
    services: the declared children.
    unit: this machine's resident unit for `root`.
    """

    @staticmethod
    def _environment(process: psutil.Process) -> set[str]:
        """The names in `process`'s environment, never their values; empty when unreadable.

        macOS hides an Apple platform binary's environment (`/bin/sh`, `/bin/sleep`) and answers
        an empty one, so an empty answer counts as unread: every child holds the allowlist.
        """
        try:
            return set(process.environ())
        except psutil.Error:
            return set()

    @staticmethod
    def _environments(tree: Sequence[psutil.Process]) -> list[set[str]]:
        """The names of each environment of `tree` this user may read."""
        return [names for names in map(Supervisor._environment, tree) if names]

    @staticmethod
    def _exposed(tree: Sequence[psutil.Process]) -> list[str]:
        """Every address a process of `tree` listens on that is not loopback."""
        return [address for process in tree for address in Supervisor._listening(process)]

    @staticmethod
    def _listening(process: psutil.Process) -> Iterator[str]:
        """The addresses `process` listens on off loopback; none for one gone or closed to us."""
        with suppress(psutil.Error):
            for connection in process.net_connections(kind="inet"):
                address = ipaddress.ip_address(connection.laddr.ip)
                if connection.status == psutil.CONN_LISTEN and not address.is_loopback:
                    yield f"{connection.laddr.ip}:{connection.laddr.port}"

    def __init__(
        self,
        root: Path,
        *,
        programs: Programs,
        unit: Unit,
        services: Sequence[Service] = SERVICES,
    ) -> None:
        self.root = root
        self.programs = programs
        self.services = services
        self.unit = unit
        self.layout = Layout(Project().out(root) / "serve")
        self.ledger = Ledger(self.layout.ledger)
        self.client = Client(self.layout.socket)
        self.dotenv = frozenset(Dotenv.at(root).values)
        self.compose = Compose(
            services, root=root, layout=self.layout, programs=programs, dotenv=self.dotenv
        )

    def _absent(self) -> Section:
        """The row while nothing answers on the socket."""
        if self.unit.loaded():
            return Section(
                section="supervisor",
                verdict=Verdict.FAIL,
                detail=(
                    f"{self.unit.name} is loaded but nothing answers on {self.layout.socket}; "
                    f"read {self.layout.log}"
                ),
                fix=f"{Project().name} serve",
            )
        if left := self.ledger.members():
            return Section(
                section="supervisor",
                verdict=Verdict.FAIL,
                detail=f"{len(left)} processes of a stopped serve are still running",
                fix=f"{Project().name} serve stop",
            )
        missing = (
            ""
            if self.programs.binaries.find(_SUPERVISOR)
            else f", and {_SUPERVISOR} is not installed here"
        )
        return Section(
            section="supervisor",
            verdict=Verdict.WARN,
            detail=f"not running{missing}; the owner starts it",
            fix=f"{Project().name} serve",
        )

    def _answering(self, supervised: ProjectState) -> Section:
        """The supervisor's row while it answers: its age, the unit, the socket and the leaks.

        It fails when the socket is not 0600, when a process of the tree holds a name the
        workspace `.env` defines, or when one listens off loopback. A supervisor no unit holds
        is a word: nothing would start it again.
        """
        tree = self.ledger.members()
        environments = Supervisor._environments(tree)
        leaked = sorted(set().union(*environments) & self.dotenv)
        exposed = Supervisor._exposed(tree)
        faults = [
            *([] if self.layout.guarded() else ["the socket is not 0600"]),
            *([f"{', '.join(leaked)} from .env reach a child"] if leaked else []),
            *([f"listening on {', '.join(exposed)}"] if exposed else []),
        ]
        if faults:
            return Section(
                section="supervisor",
                verdict=Verdict.FAIL,
                detail=f"{'; '.join(faults)} (process-compose {supervised.version})",
                fix=f"{Project().name} serve stop",
            )
        kept = self.ledger.keeper()
        loaded = self.unit.loaded()
        uptime = timedelta(seconds=supervised.up_time // NANOSECONDS_PER_SECOND)
        return Section(
            section="supervisor",
            verdict=Verdict.PASS if loaded else Verdict.WARN,
            detail=(
                f"process-compose {supervised.version} up {uptime}, "
                f"keeper pid {kept.pid if kept else '?'}, {len(tree)} processes, unit "
                f"{self.unit.name} {'loaded' if loaded else 'NOT loaded'}; socket 0600, no .env "
                f"name in the {len(environments)} environments readable, nothing listening "
                "off loopback"
            ),
            fix="" if loaded else f"{Project().name} serve",
        )

    def _awaited(self) -> None:
        """Wait until the started unit's supervisor answers, recorded and its socket guarded.

        process-compose binds at 0700 and the keeper narrows it within a tick, so the first
        report never shows the moment between. Refused past `_STARTING`, naming the logs.
        """
        deadline = time.monotonic() + _STARTING
        while not (self.client.alive() and self.ledger.supervisor() and self.layout.guarded()):
            if time.monotonic() > deadline:
                raise MissionError(
                    f"{self.unit.name} did not answer on {self.layout.socket} within "
                    f"{_STARTING:.0f}s; read {self.layout.log} and {self.layout.unit_log}"
                )
            time.sleep(_POLL)

    def _child(self, service: Service, processes: Mapping[str, Process] | None) -> Section:
        """One declared child's row: off with its reason, or what the supervisor says of it."""
        row = Section(section=service.name, verdict=Verdict.PASS, detail=f"off: {service.off}")
        if service.off:
            return row
        if processes is None:
            return row.model_copy(
                update={
                    "verdict": Verdict.WARN,
                    "detail": "unknown: the supervisor is not running",
                }
            )
        if (process := processes.get(service.name)) is None:
            return row.model_copy(
                update={
                    "verdict": Verdict.FAIL,
                    "detail": "declared, but the running YAML is older and lacks it",
                    "fix": f"{Project().name} serve",
                }
            )
        if service.kind is Kind.PERIODIC:
            return self._periodic(row, service, process)
        return self._resident(row, service, process)

    def _command(self, supervisor: Path) -> list[str]:
        """The unit's argv: the keeper by path, then process-compose under the allowlist alone."""
        return [
            sys.executable,
            "-I",
            getfile(keeper),
            str(self.layout.state),
            "--",
            *self._isolated(supervisor),
            "-U",
            "--unix-socket",
            str(self.layout.socket),
            "-t=false",
            "--disable-dotenv",
            "-L",
            str(self.layout.supervisor_log),
            "-f",
            str(self.layout.compose),
        ]

    def _halt(self) -> int:
        """Remove the unit, wait while its keeper stops the tree, and sweep whatever is left.

        Answers how many processes had to be swept. The service manager answers a removal at
        once while the supervisor stops its children one after another, so the wait is ours;
        once the keeper is gone nothing else would stop a child, so the sweep follows at once.
        """
        self.unit.remove()
        deadline = time.monotonic() + self.compose.stop_seconds() + _MARGIN
        while self._keeping() and time.monotonic() < deadline:
            time.sleep(_POLL)
        if (kept := self.ledger.keeper()) is not None and kept.alive():
            with suppress(ProcessLookupError):
                os.kill(kept.pid, signal.SIGKILL)
        swept = self.ledger.sweep()
        self.layout.socket.unlink(missing_ok=True)
        return swept

    def _isolated(self, supervisor: Path) -> list[str]:
        """process-compose started from an empty environment holding the allowlist alone."""
        allowed = [f"{name}={value}" for name, value in Compose.allowed().items()]
        return ["/usr/bin/env", "-i", *allowed, str(supervisor)]

    def _keeping(self) -> bool:
        """Whether the keeper the ledger names still runs."""
        kept = self.ledger.keeper()
        return kept is not None and kept.alive()

    def _periodic(self, row: Section, service: Service, process: Process) -> Section:
        """A pass's row: when it runs next, and a warning naming its log after a failed run."""
        upcoming = process.next_run_time
        next_run = upcoming.astimezone().strftime("%H:%M:%S") if upcoming else "unscheduled"
        log = self.layout.logs / f"{service.name}.log"
        failed = f", the last pass exited {process.exit_code}; read {log}"
        return row.model_copy(
            update={
                "verdict": Verdict.WARN if process.exit_code else Verdict.PASS,
                "detail": (
                    f"every {service.every}, next at {next_run}"
                    f"{failed if process.exit_code else ''}"
                ),
            }
        )

    def _prepared(self) -> Path:
        """The supervisor's executable, once the state directory is ready for it.

        Refuses a missing or unfit process-compose, and a socket path past `sun_path`, which
        process-compose meets with `bind: invalid argument` and no child started.
        """
        found = self.programs.binaries.check(_SUPERVISOR, wanted=_WANTED).path
        size = len(os.fsencode(self.layout.socket))
        if size >= _SOCKET_BYTES:
            raise MissionError(
                f"{self.layout.socket} is {size} bytes, past the {_SOCKET_BYTES - 1} a unix "
                "socket path holds; serve a workspace whose path is shorter"
            )
        self.layout.state.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.layout.state.chmod(0o700)
        return found

    def _replies(self) -> tuple[ProjectState, dict[str, Process]]:
        """The supervisor's own state and its children by name, Unanswered when it is silent."""
        return self.client.project(), {child.name: child for child in self.client.processes()}

    def _resident(self, row: Section, service: Service, process: Process) -> Section:
        """A resident child's row: running or stopping, or failed once past its restarts."""
        if process.is_running or process.status in _STOPPING:
            detail = f"{process.status.lower()}, pid {process.pid}, {process.restarts} restarts"
            return row.model_copy(update={"detail": detail})
        log = self.layout.logs / f"{service.name}.log"
        return row.model_copy(
            update={
                "verdict": Verdict.FAIL,
                "detail": (
                    f"{process.status.lower()}, exit {process.exit_code} after "
                    f"{process.restarts} restarts; read {log}"
                ),
                "fix": f"{Project().name} serve",
            }
        )

    def _standing(self) -> bool:
        """Whether anything of the tree still runs: the keeper, or its supervisor's session."""
        return self._keeping() or bool(self.ledger.members())

    def _validated(self, supervisor: Path) -> None:
        """Refuse a YAML this process-compose would not start, before the unit is told to."""
        said = ToolRunner(timeout=_VALIDATING, max_bytes=64 << 10).run(
            self._isolated(supervisor)[0],
            [
                *self._isolated(supervisor)[1:],
                "up",
                "--dry-run",
                "--disable-dotenv",
                "-t=false",
                "-L",
                str(self.layout.state / "dry-run.log"),
                "-f",
                str(self.layout.compose),
            ],
            cwd=self.layout.state,
        )
        if said.returncode:
            spoken = (said.stdout + said.stderr).splitlines()
            said_last = (line.strip() for line in reversed(spoken) if line.strip())
            reason = next(said_last, "it said nothing")
            raise MissionError(f"process-compose refused {self.layout.compose}: {reason}")

    def section(self) -> Section:
        """The doctor's one `serve` row: the supervisor's, failing too when a child fails."""
        head, *children = self.status()
        broken = [row.section for row in children if row.verdict is Verdict.FAIL]
        if head.verdict is Verdict.FAIL or not broken:
            return head.model_copy(update={"section": "serve"})
        return Section(
            section="serve",
            verdict=Verdict.FAIL,
            detail=f"{head.detail}; failing: {', '.join(broken)}",
            fix=f"{Project().name} serve status",
        )

    def start(self) -> list[Section]:
        """Replace whatever runs with the registry's children under a fresh unit, then report.

        Nothing is installed until the YAML is written and process-compose accepts it.
        """
        supervisor = self._prepared()
        self._halt()
        self.compose.write()
        self._validated(supervisor)
        self.unit.install(
            self._command(supervisor),
            stop_seconds=self.compose.stop_seconds(),
            log=self.layout.unit_log,
        )
        self._awaited()
        return self.status()

    def status(self) -> list[Section]:
        """The supervisor's row, then one row per declared child, in the registry's order."""
        try:
            supervised, processes = self._replies()
        except Unanswered:
            return [self._absent(), *(self._child(service, None) for service in self.services)]
        children = [self._child(service, processes) for service in self.services]
        return [self._answering(supervised), *children]

    def stop(self) -> list[Section]:
        """Remove the unit and wait until nothing of the tree runs, sweeping what outstays it."""
        swept = self._halt()
        if self._standing():
            return [
                Section(
                    section="supervisor",
                    verdict=Verdict.FAIL,
                    detail=f"{len(self.ledger.members())} processes survived the stop",
                    fix=f"{Project().name} serve stop",
                )
            ]
        outstayed = f"; swept {swept} processes that outstayed it" if swept else ""
        return [
            Section(
                section="supervisor",
                verdict=Verdict.PASS,
                detail=f"{self.unit.name} removed, and nothing of it runs{outstayed}",
            )
        ]
