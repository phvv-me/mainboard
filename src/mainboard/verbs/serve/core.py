from ...core.services import resident
from ...engines.compile.ecosystems import Binary
from ...engines.compile.provisioner import Provisioner
from ...manifest.loading import load
from ...policy.warden import Warden
from ...render import Output
from ...serve.compose import Programs
from ...serve.constants import SERVICES
from ...serve.supervisor import Supervisor
from ..constants import TREE
from ..options import sectioned
from ..session import Verbs

_COMPACT = Output()


class ServeVerbs(Verbs):
    """The `serve` verbs: the owner's to type, since what they start runs unattended.

    Every one refuses when an agent runs it (`Scope.OWNER` in TREE): an agent that installed the
    unit would start the very children that act without it.
    """

    def _supervisor(self) -> Supervisor:
        """This workspace's supervisor, its programs found as `binary path` finds them."""
        root = self.session.workspace_root()
        stage = Provisioner(root, load(self.session.project.manifest(root))).shard().stage
        programs = Programs.here(Binary.of(stage, "default").resolver())
        return Supervisor(root, programs=programs, unit=resident(root))

    def serve(self, *, output: Output = _COMPACT) -> int:
        """Start every declared service under one supervisor this machine keeps running.

        One user unit (launchd over `ssh -tt localhost`, or systemd) runs the keeper, which runs
        process-compose on a 0600 socket from an empty environment, so no child holds a `.env`
        secret. Whatever `serve` already runs is stopped first, and nothing starts while an
        agent child it would run lacks its harness's own enforcer. Exits 1 when a row fails.
        """
        root = self.session.workspace_root()
        Warden(root, load(self.session.project.manifest(root)).policy).launch(SERVICES)
        return sectioned(self._supervisor().start(), output, title="serve")

    def status(self, *, output: Output = _COMPACT) -> int:
        """Show the supervisor and each declared service: running, scheduled, failed or off.

        Exits 1 when a row fails: the supervisor not answering while its unit is loaded, a
        socket not 0600, a `.env` name in a child's environment, a listener off loopback, or a
        resident service that stopped.
        """
        return sectioned(self._supervisor().status(), output, title="serve status")

    def stop(self, *, output: Output = _COMPACT) -> int:
        """Stop every service and remove the unit, waiting until none of its processes is left.

        It stays stopped across logins and reboots until `serve` runs again. Exits 1 when a
        process outlived the stop.
        """
        return sectioned(self._supervisor().stop(), output, title="serve stop")


_SERVE = ServeVerbs()
serve = TREE.node("serve").app(serve=_SERVE.serve, status=_SERVE.status, stop=_SERVE.stop)
