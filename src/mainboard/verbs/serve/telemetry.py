"""The telemetry verbs of `serve`: what the agents did and used, and the children collecting it."""

from datetime import UTC, datetime
from enum import StrEnum, auto
from pathlib import Path

from ...constants import REAP_DELETES
from ...core.errors import Unreachable
from ...engines.compile.ecosystems import Binary
from ...engines.compile.provisioner import Provisioner
from ...log import logger
from ...manifest.loading import load
from ...memory.outbox import Memory, Outbox
from ...render import Output
from ...serve.events.collect import Collector
from ...serve.events.drain import Drain
from ...serve.events.sink import Sink
from ...settling.reaping.reaper import Reaper
from ...state.telemetry import TelemetryLake
from ..constants import TREE
from ..options import titled
from ..session import Verbs

_COMPACT = Output()
_PENDING = ("file", "state", "lines", "oldest", "age_s")
_TURNS_SQL = "SELECT * FROM turns WHERE session = ? ORDER BY answered"
_SESSIONS_SQL = (
    "SELECT * FROM sessions WHERE latest > now()::TIMESTAMP - to_days(?) ORDER BY latest DESC"
)
_STANDINGS_SQL = "SELECT * FROM standings ORDER BY harness, span"
_SPEND_SQL = (
    "SELECT * FROM spend WHERE day > current_date - to_days(?) ORDER BY day DESC, harness, model"
)


class Child(StrEnum):
    """A child of `serve` this tool runs itself."""

    COLLECT = auto()
    DRAIN = auto()
    REAP = auto()
    SINK = auto()


class TelemetryVerbs(Verbs):
    """Read the telemetry lake, and run the children that fill it."""

    @staticmethod
    def _kept(root: Path) -> int:
        """Deliver the keeps `memory keep` queued; none while memory does not answer."""
        outbox = Outbox(root)
        if not outbox.waiting():
            return 0
        try:
            return Memory(outbox, Memory.aizk()).drain()
        except Unreachable as fault:
            logger.warning(f"{fault}; the keeps wait in the spool")
            return 0

    def _lake(self) -> TelemetryLake:
        return TelemetryLake(root=self.session.workspace_root())

    def pending(self, *, output: Output = _COMPACT) -> None:
        """Show the spool lines no pass has delivered to the sink yet, by file.

        `age_s` counts from a file's oldest waiting event. A file `refused` holds a line that is
        no spool line, so the collect pass set it aside and never sends it.
        """
        now = datetime.now(UTC)
        rows = [file.row(at=now) for file in self._lake().pending()]
        output.print_rows(rows, title="pending", columns=_PENDING)

    def process(self, name: Child) -> None:
        """Run one of serve's children in this process, as the supervisor does.

        `sink` becomes the telemetry sink, its binary found as `binary path` finds it; `collect`
        delivers mb's spooled events to it once; `drain` does as well, but ends clean, every line
        kept, while the sink is down, and delivers the keeps `memory keep` queued; `reap` looks
        for rentals this center created and nothing owns, a dry run that only logs until
        `REAP_DELETES` is set. The generated process-compose file runs them.

        Args:
            name: the child.
        """
        root = self.session.workspace_root()
        match name:
            case Child.SINK:
                provisioner = Provisioner(root, load(self.session.project.manifest(root)))
                Sink(root).start(Binary.of(provisioner.shard().stage, "default"))
            case Child.COLLECT:
                sent = Collector(Sink(root)).collect()
                logger.info(f"delivered {sent} spooled events to the sink")
            case Child.DRAIN:
                sent = Drain(Sink(root)).drain()
                logger.info(f"drained {sent} spooled events into the sink")
                logger.info(f"drained {self._kept(root)} queued keeps into memory")
            case Child.REAP:
                reaped = Reaper(self.session.board("local")).once(delete=REAP_DELETES)
                logger.info(
                    f"ruled on {len(reaped.decisions)} rentals, acted on {len(reaped.acted)}"
                )

    def sessions(self, session: str = "", *, days: int = 7, output: Output = _COMPACT) -> None:
        """Show the agent sessions of the last days, or one session's turns.

        Sessions come newest first. Telemetry is an estimate that may lose a few seconds;
        `cost_usd` is Claude's API-equivalent figure, and Codex reports tokens alone.

        Args:
            session: a session as `<harness>:<id>` to list its turns instead.
            days: how many days back the listing reaches.
        """
        if session:
            output.print_rows(
                self._lake().rows(_TURNS_SQL, [session]), title=f"turns of {session}"
            )
            return
        output.print_rows(
            self._lake().rows(_SESSIONS_SQL, [days]), title=f"sessions of the last {days} days"
        )

    def usage(self, *, days: int = 7, output: Output = _COMPACT) -> None:
        """Show what the agents used: the last 5 hours and 7 days, then each day.

        Each harness has its two trailing windows, then a row per UTC day and model. Claude's
        `cost_usd` is an API-equivalent estimate, not a bill, and Codex reports tokens alone;
        the windows trail from now, and each vendor's own meter decides a limit.

        Args:
            days: how many UTC days the daily rows reach back.
        """
        lake = self._lake()
        titled(output, "standings")
        output.print_rows(lake.rows(_STANDINGS_SQL), title="standings")
        titled(output, "spend")
        output.print_rows(lake.rows(_SPEND_SQL, [days]), title="spend")


_TELEMETRY = TelemetryVerbs()
_BOUND = TREE.node("serve").bind(
    _TELEMETRY.pending, _TELEMETRY.process, _TELEMETRY.sessions, _TELEMETRY.usage
)
pending, process = _BOUND["pending"], _BOUND["process"]
sessions, usage = _BOUND["sessions"], _BOUND["usage"]
