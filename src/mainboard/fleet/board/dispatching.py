import shlex
from functools import partial
from typing import TYPE_CHECKING, Final

from plumbum import local as localhost

from ...core.errors import MissionError
from ...core.execution.shell import foreground
from ...dispatch.backends.base import ProviderBackend, route
from ...dispatch.dispatcher import Handle, Order
from ...dispatch.hosts.landing import Landing, renter
from ...dispatch.jobs import vocabulary
from ...dispatch.jobs.commandline import joined
from ...dispatch.mirroring.shipment import Shipment
from ...dispatch.wrapping import connection
from ...engines.compile.pixi.policy import LockPolicy
from ...engines.compile.provisioner import Provisioner
from ...jobs.targets.target import Target
from ...manifest.render.scaffold import Scaffold
from ..runs import Job, ProviderJob, Run
from .tracking import Tracking

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ...context.plan import ExecutionPlan


_SSH_FAMILY: Final = "ssh-family"


class Dispatching(Tracking):
    """Running work: here, on a declared host, or on a rental, and the runs it leaves."""

    def job(self, handle: str | int, *, host: str = "") -> Run:
        """The dispatched run `handle`, rebuilt from the dispatch cache as whichever kind it is.

        A fresh process addresses a running job as the submitting one did, with no `Handle`
        reassembled by hand. The recorded kind decides the world it returns from, so a rental
        outlives the process that started it exactly as a queued job does.

        handle: the scheduler handle or provider run id the job was dispatched under.
        host: the alias to disambiguate a handle recorded on several hosts.
        """
        record = self.dispatcher.cache.run(str(handle), host or None)
        if record.verdict in {vocabulary.Word.PREPARED, vocabulary.Word.SUBMITTING}:
            action = (
                "no create attempted; cancel if abandoned"
                if record.verdict == vocabulary.Word.PREPARED
                else "inspect the provider by this label before retrying or cancelling"
            )
            raise MissionError(
                f"creation {record.creation} has no confirmed provider handle; {action}"
            )
        bound = self.on(record.target)
        rebuilt = partial(
            Handle,
            id=record.handle,
            host=record.target,
            kind=record.kind,
            fetch_path=record.fetch_path,
        )
        destination = route(record.kind)
        if destination == _SSH_FAMILY:
            return Job(bound, rebuilt(root=bound.remote_root()))
        return ProviderJob(bound, destination(), rebuilt(root=""))

    def rented(self, backend: ProviderBackend, order: Order) -> Handle:
        """Dispatch `order` and retain the rental before any workspace provisioning.

        A rental is shipped the artifact `install` ships gold: this workstation solved the lock
        and the machine installs frozen against it. The lock vouches here, before the rental
        opens, since a refusal a minute later has already cost money. A plan bringing its own
        container skips all of it, since a prebuilt image already holds everything.

        A job spelled by file needs a workspace to ship its closure into, so a plan whose image
        is the whole environment refuses it. The order's resources (spend cap and walltime)
        bound the rental, and its watcher announces each landing stage, since a landing is
        minutes of mirror, install and provisioning that would otherwise stand silent on a
        metered box.
        """
        plan, shipment = order.plan, order.shipment
        renting = renter(backend, plan)
        if renting is None:
            if shipment.sealed:
                raise MissionError(
                    f"{shipment.spelling} is a job spelled by file, and host {plan.host!r} "
                    "runs a prebuilt image that ships no workspace for its closure; run it as "
                    "a command inside that image, or on a host that mirrors the workspace"
                )
            allocation = self.dispatcher.allocating(order, evidence="pending")
            try:
                handle = backend.submit(
                    plan, shipment.command, order.resources, allocation=allocation
                )
            finally:
                allocation.interrupted()
            return Handle(
                id=handle,
                host=plan.host,
                root="",
                kind=plan.profile.kind,
                fetch_path=shipment.fetch or None,
            )
        provisioner = Provisioner(self.root, self.manifest)
        provisioner.compiler_for(plan.env).vouch()
        artifact = tuple(provisioner.artifact_for(plan.env))
        return Landing(
            self.dispatcher,
            renting,
            order.model_copy(update={"artifact": artifact}),
            floor=self.floor,
        ).land()

    def run(
        self,
        command: Sequence[str],
        *,
        env: str = "",
        container: str = "",
        policy: LockPolicy = LockPolicy.UPDATE,
        install: bool = True,
    ) -> int:
        """Run `command` through the host's activated plan, returning its exit code.

        Local commands execute in place; remote ones use one ssh connection, a batch cluster's
        login endpoint included. A declared task name is resolved by pixi inside the generated
        workspace rather than by the shell, which makes `run test` and `run pytest -q` one verb.
        A job file target runs here only, through the runner and closure format submitted jobs
        use; remotely it needs `submit` for allocation and source transfer, while collection and
        help stay local and allocate nothing.

        command: exact command argv, or a declared task name and its arguments, or a job.
        container: a container override, `none` forcing bare.
        policy: what may happen to the lock before a local run, as `pixi run` takes it; a host
            runs what its setup installed.
        install: bring the local prefix in line with the lock first.
        """
        plan = self.plan(env=env, container=container)
        target = Target.spelled(command, self.root)
        if target is not None:
            if not self.local:
                raise MissionError(
                    "remote file targets require submission for source transfer and allocation; "
                    f"use mainboard job submit --on {plan.host} -- {target.spelling}, then "
                    "`mb job show --wait` or `mb job list`. Run collection and help locally."
                )
            shipment = self.sealed(target, plan)
            command = shipment.locally(self.root, closure=self.dispatcher.stage_listing(shipment))
        if self.local and not plan.containerized:
            return Provisioner(self.root, self.manifest).run(
                command, plan.env, policy=policy, install=install
            )
        line = self.line(joined(command), env=env, container=container)
        if self.local:
            return foreground(localhost["bash"]["-lc", line])
        with connection(self.host) as remote:
            return foreground(remote["bash"]["-lc", line])

    def scaffold(self) -> Scaffold:
        """The project generator, rendering the workspace's own templates through copier."""
        return Scaffold(self)

    def shipment(
        self, command: str, plan: ExecutionPlan, *, needs: Sequence[str] = ()
    ) -> Shipment:
        """What `command` runs and ships: a job's closure when it spells one, else the mirror.

        command: the vetted command line, a job spelled `path/to/file.py::name [args]` or not.
        plan: the resolved execution context whose environment names the import roots.
        needs: data paths declared at dispatch time, which only a job can take.
        """
        target = Target.spelled(shlex.split(command), self.root)
        if target is not None:
            return self.sealed(target, plan, needs=needs)
        if needs:
            raise MissionError(
                f"--needs belongs to a job spelled by file; {command!r} is a command and "
                "reaches the mirror as it is"
            )
        return Shipment.of_command(
            command,
            source=self.dispatcher.source(command, paths=plan.profile.sync.shipped(self.root)),
            imports=self.imports(plan),
        )
