# Landing a dispatch on a machine rented for one job: the same thing `mainboard setup` does to a
# declared host, done to a box that will exist for the next half hour.
#
# A rented container has no workspace, tool or environment, so the command comes last: mirror
# the workspace, install the tool from it, provision the environment from the lock this
# workspace already solved, pin the tree the job runs from, and only then hand the waiting
# entrypoint the line that runs it. Anything earlier is the failure this module ends, a
# `mainboard run` that reached a bare image and answered `bash: mainboard: command not found`
# while the meter ran.
#
# A rental gets no record: it is gone by the next sweep, so nothing here saves a `HostSetup`,
# checks a queue daemon or probes hardware. Its own backend still owns the log, the exit marker
# and the cancel that stops the meter.

import shlex
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from ..core.errors import MissionError
from . import vocabulary
from .backends.base import ProviderBackend
from .dispatcher import Handle
from .jobs import JobSpec
from .onboard import Bootstrap
from .rentals import Rental, handoff
from .schedulers.base import failure_reason
from .shared import Watcher, announce, logger
from .shells import PosixShell
from .snapshots import CLOSURE, Snapshots
from .sync import SyncLock
from .targets import home_of, placed
from .transport import SshTransport
from .wrapping import connection, wrap

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..context.plan import ExecutionPlan
    from .allocation import Allocation
    from .dispatcher import Dispatcher
    from .shipment import Shipment
    from .transport import Machine
    from .vocabulary import Resources


@runtime_checkable
class Renter(Protocol):
    """What a landing needs of a backend: hand a rental over, and end one.

    Structural, because renting is the `Rentable` capability a provider opts into while ending a
    rental is on every `ProviderBackend`, and a landing needs both from whichever class.
    """

    def cancel(self, handle: str) -> None:
        """Cancel `handle` on the provider, which is what stops the meter."""

    def rent(self, plan: ExecutionPlan, resources: Resources, *, allocation: Allocation) -> Rental:
        """Rent a machine for one job and return it once ssh answers on it."""


def renter(backend: ProviderBackend, plan: ExecutionPlan) -> Renter | None:
    """`backend` as the thing a landing drives, or None when this dispatch runs the raw command.

    A plan declaring its own container runs raw: its prebuilt image already carries everything
    the command needs, so the one-shot container is cheaper and simpler than a landing.
    """
    if plan.containerized or not isinstance(backend, Renter):
        return None
    return backend


class Landing:
    """One rented machine, brought up to a host this workspace can dispatch into.

    plan: the provider host's execution context, container-free by construction.
    resources: what the rental is opened under, spend cap and walltime included.
    artifact: the compiled manifest, lock and state shipped with the mirror so the machine
        installs frozen rather than solving for itself.
    watch: announces each stage as it begins.
    floor: the tool version a rental installs from an index when the workspace vendors no source.
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
        floor: str = "",
    ) -> None:
        self.dispatcher = dispatcher
        self.backend = backend
        self.plan = plan
        self.resources = resources
        self.artifact = tuple(artifact)
        self.watch = watch or announce
        self.floor = floor

    def land(self, shipment: Shipment, *, name: str = "", node: str = "") -> Handle:
        """Register a rental before provisioning it, then return the same tracked handle.

        The shared registry retains the rental if this process dies during provisioning. A
        failed landing still attempts immediate cleanup; a failed cleanup remains tracked.
        The provider must persist its returned handle before waiting for SSH. A lost create
        response leaves its labeled intent visible and requires provider-side reconciliation.

        name: the label retained by a later monitor.
        node: the research node served by the dispatch.
        """
        allocation = self.dispatcher.allocating(
            self.plan, shipment, self.resources, name=name, node=node, evidence="not_started"
        )
        rental = None
        started = False
        try:
            rental = self.backend.rent(self.plan, self.resources, allocation=allocation)
            self.equip(rental, shipment=shipment)
            started = True
        finally:
            if not started:
                if rental is None:
                    allocation.interrupted()
                else:
                    self._abort(rental.handle)
        return Handle(
            id=rental.handle,
            host=self.plan.host,
            root="",
            kind=self.plan.profile.kind,
            fetch_path=shipment.fetch or None,
        )

    def _abort(self, handle: str) -> None:
        """Release a definite setup failure; retain an ambiguous launch for the monitor."""
        registered = None
        try:
            registered = self.dispatcher.cache.run(handle, self.plan.host)
        except LookupError:
            logger.warning("registration of rental %s failed before provisioning", handle)
        if registered is not None and registered.evidence != "not_started":
            logger.warning("launch of %s is uncertain; retaining the tracked rental", handle)
            return
        try:
            if registered is not None:
                self.dispatcher.cache.resolve(
                    registered, vocabulary.FAILED, None, vocabulary.FAILED
                )
        finally:
            logger.warning("landing on %s failed before launch; ending the rental", handle)
            self.backend.cancel(handle)

    def equip(self, rental: Rental, *, shipment: Shipment) -> None:
        """Mirror, install, provision, pin, launch: everything the machine needs, in that order.

        shipment: its provenance is read once, minutes before the pin uses it, since a dirty
            tree's key digests its own delta, which could move under a second reading.
        """
        policy = SshTransport(endpoint=rental.endpoint)
        where = rental.endpoint.destination
        with (
            SyncLock(rental.endpoint, self.dispatcher.sync.root),
            connection(where, policy) as remote,
        ):
            root = placed(self.plan.profile.root, home=home_of(remote))
            listing = self.dispatcher.stage_listing(shipment)
            script = self.script(shipment, root=root, listing=listing)
            self.transferable(remote)
            self.watch(f"mirroring the workspace to {where}:{root}")
            shipped = self.dispatcher.mirror(
                self.plan,
                root,
                ssh=policy,
                required=[self.artifact] if self.artifact else [],
                extra=[script, *([listing] if listing else []), *shipment.files],
                fetch=shipment.fetch,
            )
            # Every command below stands in the workspace the mirror just created, which is why
            # nothing before this line may `cd` into a root that did not exist yet.
            bootstrap = Bootstrap(PosixShell(remote, self.plan, root), floor=self.floor)
            self.watch(f"installing the tool on {rental.handle}")
            bootstrap.tool()
            self.watch(f"provisioning {self.plan.env} on {rental.handle}")
            bootstrap.environment()
            self.watch(f"pinning the source tree on {rental.handle}")
            pinned = Snapshots(root).pin(
                self.dispatcher.agent(self.plan, ssh=policy),
                key=shipment.source.key,
                image=self.dispatcher.image(self.plan, shipment, listing=listing, shipped=shipped),
                results=shipment.fetch,
                commit=shipment.source.commit,
                digest=shipment.source.digest,
                script=script,
            )
            self.verify(remote, pinned)
            self.watch(f"starting the job on {rental.handle}")
            with self.dispatcher.cache.settlement:
                registered = self.dispatcher.cache.run(rental.handle, self.plan.host)
                if registered.verdict in vocabulary.TERMINAL:
                    raise MissionError(
                        f"rental {rental.handle} ended during setup; refusing launch"
                    )
                self.dispatcher.cache.delivery(registered, "pending")
                self.start(remote, pinned=pinned, script=f"{pinned}/{Snapshots.script(script)}")

    def transferable(self, remote: Machine) -> None:
        """Make sure the machine can receive a mirror at all, since its Python runs the far end.

        A rented image almost always ships one; one that does not gets it through apt, which
        every provider base image this house rents is built on, and an image with neither
        refuses with what the machine said, before the mirror. It runs on the bare connection,
        since the workspace a later step would `cd` into does not exist until the mirror.
        """
        probe = f"{self.plan.profile.python} -c pass"
        retcode, _, _ = remote["bash"][["-lc", probe]].run(retcode=None)
        if retcode == 0:
            return
        self.watch("installing python on the rental")
        install = "apt-get update -qq && apt-get install -y -qq python3"
        retcode, _, err = remote["bash"][["-lc", install]].run(retcode=None)
        if retcode:
            raise MissionError(
                f"the rented machine has no python and could not install one: "
                f"{str(err).strip()[-400:]}"
            )

    def verify(self, remote: Machine, pinned: str) -> None:
        """Prove the pinned tree activates before the entrypoint is asked to run a job from it.

        The last cheap moment: nothing has started, so a tree the job could not activate from
        ends the rental here instead of being paid for in full and answering with its own
        activation refusal (vast 49867368, exit 1, 2026-09-04).
        """
        line = wrap(self.plan, pinned, command="true")
        retcode, _, err = remote["bash"][["-lc", line]].run(retcode=None)
        if retcode:
            raise MissionError(
                f"the pinned tree on the rental cannot run a command: "
                f"{failure_reason(str(err), int(retcode))}"
            )

    def script(self, shipment: Shipment, *, root: str, listing: str) -> str:
        """Render the job script this rental runs and stage it for the mirror to carry.

        The same job an ssh host runs, so a rented run's receipts, walltime cap and
        `MAINBOARD_SOURCE` stamp match one measured on gold. It activates from the tree this
        dispatch is about to pin, a path computed here and materialised on the machine later.

        root: the workspace root on the machine.
        listing: the staged closure listing, workspace-relative, empty for a command.
        """
        pinned = self.dispatcher.pinned(root, source=shipment.source)
        spec = JobSpec(
            cmd=shipment.command,
            plan=self.plan,
            root=pinned,
            walltime=self.resources.walltime or "",
            gpus=self.resources.gpus,
            mem_gb=self.resources.mem_gb,
            pythonpath=":".join(f"{pinned}/{place}".rstrip("/") for place in shipment.imports),
            source=shipment.source.identity,
            commit=shipment.source.commit,
            digest=shipment.source.digest,
            closure=f"{pinned}/{CLOSURE}" if listing else "",
            first_party=":".join(shipment.first_party),
            deferred=":".join(shipment.deferred),
            exports=self.plan.exports,
        )
        return self.dispatcher.write_job_script(spec, pbs=False)

    def start(self, remote: Machine, *, pinned: str, script: str) -> None:
        """Hand the waiting entrypoint the line that runs the job from the tree that was pinned.

        A rented box has no queue, and must not: its entrypoint owns the log, the exit marker and
        the meter, so the job starts there or the run leaves no readable receipt. The line is
        the staging every host gets (`cd`, PATH, modules) around a job that activates itself.

        script: the verified frozen wrapper inside the snapshot, named absolutely so launch does
            not depend on the mirror's mutable dispatch links.
        """
        line = wrap(self.plan, pinned, command=f"sh {shlex.quote(script)}", activate=False)
        retcode, _, err = (remote["bash"]["-c", handoff()] << f"{line}\n").run(retcode=None)
        if retcode:
            raise MissionError(f"could not start the job on the rental: {str(err).strip()[-400:]}")
