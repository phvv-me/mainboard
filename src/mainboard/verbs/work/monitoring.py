from ...core.project import Project
from ...core.services import settler
from ...log import logger
from ...render import Output
from ..constants import TREE
from ..session import Verbs
from .jobs import JobVerbs

_COMPACT = Output()


class MonitorVerbs(Verbs):
    """The settling pass this machine's service manager runs, and how to pause it."""

    jobs: JobVerbs

    def monitor(self, *, output: Output = _COMPACT) -> None:
        """Settle once, as `job list --json` does, for periodic runners installed before it.

        Those runners read its `--json`; `start`, `status` and `stop` manage the installed pass.
        """
        self.jobs.jobs(output=output)

    def start(self) -> None:
        """Arm the installed settling pass again after `stop`; one already armed is left alone.

        launchd loads the agent (which runs a pass at once), systemd enables and starts the timer.
        Refuses when no pass is installed, which `job list --every 20m` does.
        """
        logger.result(f"{Project().name}: {settler(self.session.workspace_root()).start()}")

    def status(self) -> int:
        """Say whether the settling pass is installed and armed, when it last ran, and the fix.

        Exits 0 while a pass is armed and 1 otherwise, so a script can branch on it.
        """
        settling = settler(self.session.workspace_root()).state()
        logger.result(f"{Project().name}: {settling.detail}")
        if settling.fix:
            logger.result(f"{Project().name}: run `{settling.fix}`")
        return int(not settling.active)

    def stop(self) -> None:
        """Unload the settling pass but keep it installed, so no pass runs until `start`.

        Dispatched jobs stay unsettled meanwhile; nothing is lost, and the next pass settles them.
        launchd unloads the agent and keeps its plist, systemd stops the timer and its service.
        Stopping a pass that is not running says so.
        """
        logger.result(f"{Project().name}: {settler(self.session.workspace_root()).stop()}")


_VERBS = MonitorVerbs(jobs=JobVerbs())
monitor = TREE.node("job monitor").app(_VERBS.monitor, _VERBS.start, _VERBS.status, _VERBS.stop)
