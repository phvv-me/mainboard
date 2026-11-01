# The doctor's rows on what `serve` runs here: the executables pinned by release archive that the
# services run, the supervisor keeping them, the telemetry lake the sink writes, and the native
# enforcers `[policy]` checks. Each asks the subsystem that owns the answer, as every other doctor
# row does.

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from ..core import services
from ..core.errors import MissionError
from ..core.project import Project
from ..core.section import Section
from ..engines.compile.ecosystems import Binary
from ..engines.compile.provisioner import Provisioner
from ..enums import Verdict
from ..policy.enforcers import Enforcement

if TYPE_CHECKING:
    from ..fleet.board import Board


class Serving:
    """The `binaries`, `memory`, `serve`, `telemetry` and `policy` rows of one doctor report.

    board: the workspace examined.
    env: the environment examined, the board profile's own when empty.
    """

    def __init__(self, board: Board, *, env: str = "") -> None:
        self.board = board
        self.env = env

    def _binary(self) -> Binary:
        """The executables of the environment this report examines: its pins and its resolver."""
        environment = self.board.plan(env=self.env, container="none").env
        shard = Provisioner(self.board.root, self.board.manifest).shard(environment)
        return Binary.of(shard.stage, environment)

    def binaries(self) -> Section | None:
        """Whether every tool this platform pins by release archive is installed at its pin.

        A missing or altered executable is broken, since the pin is what the services that run
        it were written against; a platform that pins none has no row.
        """
        return self._binary().section()

    def memory(self) -> Section:
        """Whether any keep waits in the spool that no pass has delivered to memory.

        It reads the spool alone and asks no server, so a memory that is down cannot hang a doctor.
        """
        # Imported here alone, for the reason `serve` gives.
        from ..memory.health import MemoryHealth
        from ..memory.outbox import Outbox

        return MemoryHealth.of(Outbox(self.board.root)).section(datetime.now(UTC))

    def policy(self) -> Section:
        """The state of each native enforcer `[policy]` checks, listed in one row.

        Those are the agent children's harness controls, the supervisor's allowlist, the
        providers' caps, N2's deny rules and the owner's grants file.
        """
        # Imported here alone, for the reason `serve` gives.
        from ..serve.compose import Compose
        from ..serve.constants import SERVICES

        board = self.board
        return Enforcement(board.root, SERVICES).section(
            board.manifest.declared(), allowed=Compose.allowed().keys(), at=datetime.now(UTC)
        )

    def serve(self) -> Section:
        """Whether `serve`'s supervisor runs here, and runs as the safety rules require.

        Not running is a word, since only the owner starts it. Running, it fails on a socket that
        is not 0600, a `.env` name in any of its processes' environments, a listener off loopback
        or a resident service that stopped, all read from the processes themselves.
        """
        # Imported here alone while the tool runs Python 3.14, which ignores `__lazy_modules__`:
        # the library facade imports the doctor through `fleet.board`, and `mb --help` imports no
        # serve module (ROADMAP-v2 W1 gate). On 3.15 these join the module's lazy imports.
        from ..serve.compose import Programs
        from ..serve.supervisor import Supervisor

        root = self.board.root
        programs = Programs.here(self._binary().resolver())
        try:
            return Supervisor(root, programs=programs, unit=services.resident(root)).section()
        except MissionError as fault:
            return Section(
                section="serve",
                verdict=Verdict.FAIL,
                detail=str(fault),
                fix=f"{Project().name} serve status",
            )

    def telemetry(self) -> Section:
        """Whether this environment's DuckDB still reads the lake `serve`'s sink writes."""
        # Imported here alone, for the reason `serve` gives.
        from ..serve.events.sink import Sink

        return Sink(self.board.root).section(self._binary().resolver())
