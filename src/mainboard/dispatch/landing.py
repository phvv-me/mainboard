# Landing a dispatch on a machine rented for one job: the same thing `mainboard setup` does to a
# declared host, done to a box that will exist for the next half hour.
#
# The order is not a matter of taste. A rented container has no workspace, no tool and no
# environment, so the command has to be the last thing that happens: mirror the workspace onto it,
# install the tool from that mirror, provision the environment from the lock this workspace
# already solved, pin the tree the job runs from, and only then hand the waiting entrypoint the
# line that runs it. Anything earlier is the failure this module exists to end, a `mainboard run`
# that reached a bare image and answered `bash: mainboard: command not found` while the meter ran.
#
# What a rental does not get is a record. A declared host is onboarded once and remembered, and
# this machine is gone by the next sweep, so nothing here saves a `HostSetup`, checks a queue
# daemon it will never dispatch through, or probes hardware nobody will read back. The rental
# answers for itself exactly as it did before, through its own backend: the log, the exit marker
# and the cancel that stops the meter are untouched by any of this.

import shlex
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from ..core.errors import MissionError
from .backends.base import ProviderBackend
from .dispatcher import source_of
from .jobs import JobSpec
from .onboard import Bootstrap, RemoteShell, Watcher, announce
from .rentals import Rental, handoff
from .shared import logger
from .snapshots import Snapshots, source_key
from .targets import find_root
from .transport import SshTransport
from .wrapping import connection, wrap

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..context.plan import ExecutionPlan
    from .dispatcher import Dispatcher
    from .transport import Machine
    from .vocabulary import Resources


@runtime_checkable
class Renter(Protocol):
    """What a landing needs of a backend: hand a rental over, and end one.

    A structural pair rather than a class, because its two halves are declared in different
    places by design. Renting is the `Rentable` capability a provider opts into, ending a rental
    is on every `ProviderBackend` there is, and a landing needs both without caring which class
    either arrived from.
    """

    def cancel(self, handle: str) -> None:
        """Cancel `handle` on the provider, which is what stops the meter."""

    def rent(self, plan: ExecutionPlan, resources: Resources) -> Rental:
        """Rent a machine for one job and return it once ssh answers on it."""


def renter(backend: ProviderBackend, plan: ExecutionPlan) -> Renter | None:
    """`backend` as the thing a landing drives, or None when this dispatch runs the raw command.

    A backend that hands out ssh gets the workspace, the tool and the environment, since a bare
    rental holds none of the three. A plan that declares its own container is the one exception:
    a prebuilt image already carries everything its command needs, so there is nothing to install
    and the one-shot container is both cheaper and simpler.

    backend: the provider backend this dispatch resolved to.
    plan: the resolved execution context, whose container decides the shape.
    """
    if plan.containerized or not isinstance(backend, Renter):
        return None
    return backend


class Landing:
    """One rented machine, brought up to a host this workspace can dispatch into.

    dispatcher: the dispatch core whose mirror, job staging and workspace root the landing uses.
    backend: the provider backend that rents the machine and later settles and cancels the run.
    plan: the resolved execution context for the provider host, container-free by construction.
    resources: the resource request the rental is opened under, spend cap and walltime included.
    artifact: the compiled manifest, lock and state that ship with the mirror so the machine can
        install frozen rather than solving for itself.
    watch: announces each stage as it begins.
    """

    def __init__(
        self,
        dispatcher: Dispatcher,
        backend: Renter,
        plan: ExecutionPlan,
        *,
        resources: Resources,
        artifact: Sequence[str] = (),
        watch: Watcher | None = None,
    ) -> None:
        self.dispatcher = dispatcher
        self.backend = backend
        self.plan = plan
        self.resources = resources
        self.artifact = tuple(artifact)
        self.watch = watch or announce

    def land(self, command: str) -> str:
        """Rent a machine, land this workspace on it, start `command`, hand back the handle.

        The rental is ended the moment anything at all goes wrong, because between the create and
        the launch this process is the only thing that holds the handle: the machine's own
        entrypoint would otherwise wait out its deadline on the meter for a dispatch that already
        failed.

        command: the command the job runs, once the machine can run it.
        """
        rental = self.backend.rent(self.plan, self.resources)
        started = False
        try:
            self.equip(rental, command=command)
            started = True
        finally:
            if not started:
                logger.warning("landing on %s failed; ending the rental", rental.handle)
                self.backend.cancel(rental.handle)
        return rental.handle

    def equip(self, rental: Rental, *, command: str) -> None:
        """Mirror, install, provision, pin, launch: everything the machine needs, in that order.

        rental: the machine the provider just handed over, its entrypoint waiting.
        command: the command the job runs.
        """
        policy = SshTransport(endpoint=rental.endpoint)
        where = rental.endpoint.destination
        source = source_of(command, self.dispatcher.root)
        with connection(where, policy) as remote:
            root = self.plan.profile.root or find_root(remote)
            script = self.script(command, root=root, source=source)
            self.transferable(remote)
            self.watch(f"mirroring the workspace to {where}:{root}")
            shipped = self.dispatcher.rsync_up(
                self.plan,
                root,
                ssh=policy,
                required=[self.artifact] if self.artifact else [],
                extra=[script],
            )
            # Every command below stands in the workspace the mirror just created, which is why
            # nothing before this line may `cd` into a root that did not exist yet.
            bootstrap = Bootstrap(RemoteShell(remote, self.plan, root))
            self.watch(f"installing the tool on {rental.handle}")
            bootstrap.tool()
            self.watch(f"provisioning {self.plan.env} on {rental.handle}")
            bootstrap.environment()
            self.watch(f"pinning the source tree on {rental.handle}")
            pinned = Snapshots(root).pin(
                remote,
                key=source_key(self.dispatcher.root, source=source),
                sources=shipped,
                filters=self.dispatcher.sync.filters,
                exclude=[*self.dispatcher.sync.excludes, *self.plan.profile.sync.exclude],
            )
            self.watch(f"starting the job on {rental.handle}")
            self.start(remote, pinned=pinned, script=script)

    def transferable(self, remote: Machine) -> None:
        """Make sure the machine can receive a mirror at all, since rsync runs on both ends.

        A declared host has rsync because whoever set it up installed one. A rented image often
        ships none, and a missing far-side rsync fails the transfer with `rsync: command not
        found` on a box we already own outright, so it is installed here through the package
        manager every provider base image this house rents is built on. An image carrying neither
        refuses with what the machine itself said, before the mirror rather than during it.

        This runs on the bare connection rather than through the workspace shell every later step
        uses, because the workspace does not exist yet: the mirror below is what creates it.

        remote: the open connection to the machine.
        """
        found, _, _ = remote["bash"][["-lc", "command -v rsync"]].run(retcode=None)
        if not found:
            return
        self.watch("installing rsync on the rental")
        install = "apt-get update -qq && apt-get install -y -qq rsync"
        retcode, _, err = remote["bash"][["-lc", install]].run(retcode=None)
        if retcode:
            raise MissionError(
                f"the rented machine has no rsync and could not install one: "
                f"{str(err).strip()[-400:]}"
            )

    def script(self, command: str, *, root: str, source: str) -> str:
        """Render the job script this rental runs and stage it for the mirror to carry.

        The same bash script an ssh host runs, which is what makes a rented run's receipts, its
        walltime cap and its `MAINBOARD_SOURCE` stamp identical to one measured on gold. It
        activates from the tree this dispatch is about to pin rather than from the mirror, so the
        path is arithmetic here and materialised on the machine a few lines later.

        command: the command the job runs.
        root: the workspace root on the machine.
        source: the dispatching tree's identity, as the job's receipts carry it.
        """
        spec = JobSpec(
            cmd=command,
            plan=self.plan,
            root=self.dispatcher.pinned(root, source=source),
            walltime=self.resources.walltime or "",
            gpus=self.resources.gpus,
            mem_gb=self.resources.mem_gb,
            source=source,
            exports=self.plan.exports,
        )
        return self.dispatcher.write_job_script(spec, pbs=False)

    def start(self, remote: Machine, *, pinned: str, script: str) -> None:
        """Hand the waiting entrypoint the line that runs the job from the tree that was pinned.

        A rented box has no queue to submit to, and must not have one: its own entrypoint owns the
        log, the exit marker and the meter, so the job starts there or the run has no receipt
        anyone can read afterwards. The line is the staging every other host gets, `cd`, PATH and
        modules, around a script that does its own activation.

        remote: the open connection to the machine.
        pinned: the snapshot the job runs from.
        script: the workspace-relative job script the mirror carried over.
        """
        line = wrap(self.plan, pinned, command=f"bash {shlex.quote(script)}", activate=False)
        retcode, _, err = (remote["bash"]["-c", handoff()] << f"{line}\n").run(retcode=None)
        if retcode:
            raise MissionError(f"could not start the job on the rental: {str(err).strip()[-400:]}")
