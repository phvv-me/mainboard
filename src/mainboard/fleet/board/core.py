import json
import time
from typing import TYPE_CHECKING

from ...batch.estimate.core import Estimator, JobEstimate
from ...batch.estimate.transfer import TransferSet
from ...batch.runner import Batch
from ...batch.spec import BatchJob, Selection
from ...batch.watch import Watch
from ...constants import SSH_FAMILY
from ...context.admission import admit
from ...core.errors import MissionError
from ...core.section import Section
from ...deps import Dependencies
from ...dispatch.backends.base import route
from ...dispatch.dispatcher import Handle, Order
from ...dispatch.jobs.commandline import vetted
from ...dispatch.jobs.vocabulary import Request
from ...dispatch.transport.shells import open_shell
from ...engines.compile.provisioner import Provisioner
from ...experiments.identity import run_id
from ...runtime.job import walltime_seconds
from ...settling.monitor import Monitor
from ...tracking import streamed
from ...upkeep import machine as upkeep
from ...upkeep.doctor import Doctor
from ...waiting.verdicts import Verdicts
from ..compute import Survey
from ..runs import Job, ProviderJob, Run
from .dispatching import Dispatching

if TYPE_CHECKING:
    from ...batch.spec import BatchSpec
    from ...dispatch.shared import Watcher


class Board(Dispatching):
    """The one addressable interface: a workspace, pivoted onto a host by `on`.

    `Board()` finds the manifest like git finds a repository. The unbound board is this machine;
    `board.on("gold")` is the same board bound to a declared host, where `run`, `submit` and
    `facts` keep their shapes while the profile decides scheduler, environment, container and
    queue policy. The composed subsystems stay public for anything the facade does not carry.
    """

    def _order(self, asked: Request, *, watch: Watcher | None) -> Order:
        """`asked` resolved for this host: its plan, shipment, results path, resources and label.

        Every dispatch is named here, content-addressed over the target, the command and this
        instant, so the receipts stream has a durable key and `job list` reads better for it.
        """
        # Before any plan or transport: a command a shell cannot run costs a scheduler round
        # trip on owned hardware and a whole rental on a metered one, billed from boot.
        command = vetted(asked.command)
        plan = self.plan(env=asked.env, container=asked.container)
        shipment = self.shipment(command, plan, needs=asked.needs)
        shipment.admit(self.root)
        fetch = self.results(asked.fetch or shipment.fetch, node=asked.node, command=command)
        fingerprint = run_id({"host": plan.host, "command": shipment.spelling, "at": time.time()})
        return Order(
            plan=plan,
            shipment=shipment.model_copy(update={"fetch": fetch}),
            resources=self.resources(asked.resources(), plan=plan),
            name=asked.name or f"{plan.host}-{fingerprint[:8]}",
            node=asked.node,
            watch=watch,
        )

    def _queued(self, order: Order) -> Handle:
        """Dispatch `order` to this ssh-family host and answer its handle.

        Its environment is addressed, and its run sampled and attested.
        """
        plan, root = order.plan, self.remote_root()
        tracked = streamed(order.name, handle="")
        provisioner = Provisioner(self.root, self.manifest)
        # Before the address is taken and the mirror leaves, so the pin, the shipped artifact
        # and the manifest this command was invoked under are one thing.
        provisioner.recompiled(plan.env)
        return self.dispatcher.run(
            order.model_copy(
                update={
                    "root": root,
                    "containerize": self._containerizer(plan, root),
                    "sampler": self.sampling(tracked, resources=order.resources),
                    "attestation": self.attesting(tracked),
                    "prefix": self._addressed(plan, root),
                    "artifact": tuple(provisioner.artifact_for(plan.env)),
                }
            )
        )

    def audit(self) -> list[Section]:
        """What this host could and should update, as its own tool judges it.

        It also says whether its environments' CUDA libraries load from the environments or from
        the host.
        """
        if self.local:
            try:
                libraries = self.doctor().library_rows()
            except MissionError:
                libraries = []  # a manifest that will not load is the doctor's row to report
            return [*upkeep.audit(), *libraries]
        with open_shell(self.plan(container="none"), self.remote_root()) as shell:
            said = shell.run(f"{self.project.name} host list local --audit --json")
        start = said.find("[")
        if start < 0:
            raise MissionError(f"{self.host} answered no audit: {said.strip()[-240:]}")
        return [Section.model_validate(row) for row in json.loads(said[start:])]

    def batch(self, spec: BatchSpec, *, selection: Selection | None = None) -> Batch:
        """The declared batch over this workspace, ready to prepare, price and dispatch.

        Host-independent, since a batch names a target per job and fans across the fleet.

        selection: which of the plan's jobs to act on, all when None. The batch keeps its
            identity and receipts stream either way, so a plan sent out in waves is one batch.
        """
        return Batch(self, spec, bus=self.receipts(spec.batch_id), selection=selection)

    def compute(self) -> Survey:
        """The host-independent survey of every compute path this workspace reaches, here too."""
        return Survey(self)

    def deps(self) -> Dependencies:
        """The manifest's declared requirements, editable and re-solvable; host-independent."""
        return Dependencies(self)

    def dispatch(self, asked: Request, *, watch: Watcher | None = None) -> Run:
        """Dispatch the job `asked` describes on the host it names, and return it as a run.

        A command line ships the mirror; a job spelled `path/to/file.py::name` ships its closure
        and runs through the runner. Both are one `Shipment`, read once, so everything below
        agrees on what runs from which tree. Unset resources come from `resources`. A request
        held by a quota is asked for again through here too, so the durable sweep's retry is the
        batch's dispatch and not a second spelling of it.

        A provider host is dispatched through its backend and recorded in the same dispatch
        cache. Monitor must run to collect evidence and request release, so install its
        periodic pass for unattended jobs; provider outages can delay release, and a command
        exit or local timeout alone does not stop billing.

        Every dispatch is tracked, so an unnamed run is named here: its stream needs a key that
        outlives this process, since the settling sweep may be a cron on another day, and the
        run registry keeps exactly one such field.

        watch: announces the far-side stages long enough to be worth saying: every step of a
            rental's landing, and the priming of a queued host's environment.
        """
        if asked.target != self.host:
            return self.on(asked.target).dispatch(asked, watch=watch)
        order = self._order(asked, watch=watch)
        destination = route(order.plan.profile.kind)
        if destination != SSH_FAMILY:
            backend = destination()
            run: Run = ProviderJob(self.on(order.plan.host), backend, self.rented(backend, order))
        else:
            run = Job(self, self._queued(order))
        self.announce(
            order.name, run, command=order.shipment.spelling, host=order.plan.host, node=asked.node
        )
        return run

    def doctor(self, env: str = "") -> Doctor:
        """One verdict over this workspace and one resolved environment.

        env: the environment name, the bound host profile's own when empty.
        """
        return Doctor(self, env=env)

    def expectation(self, asked: Request) -> JobEstimate:
        """What dispatching `asked` is expected to cost, admitted and priced before dispatch.

        The resource resolution `dispatch` runs, then the queue policy check, so a refused request
        dies here in one sentence rather than after an ssh round trip. The price is the
        estimator's (a provider's metered rate, zero on owned hardware), with the declared
        walltime standing in for the runtime as a batch spec's `runtime_s` does. Nothing
        connects, rents or dispatches.
        """
        if asked.target != self.host:
            return self.on(asked.target).expectation(asked)
        plan = self.plan()
        resources = self.resources(asked.resources(), plan=plan)
        admit(
            plan.profile,
            queue=resources.queue or "",
            walltime=resources.walltime or "",
            mem_gb=resources.mem_gb or 0,
        )
        job = BatchJob(
            name=self.host,
            target=self.host,
            command=asked.command,
            runtime_s=walltime_seconds(resources.walltime) if resources.walltime else 0.0,
            queue=resources.queue or "",
            walltime=resources.walltime or "",
            mem_gb=resources.mem_gb or 0,
            gpus=resources.gpus,
            gpu_name=resources.gpu_name,
            max_usd=resources.max_usd,
            spot=resources.spot,
            arch=resources.arch,
        )
        return Estimator(self).row(job, TransferSet(job=self.host, target=self.host))

    def monitor(self) -> Monitor:
        """The host-independent durable sweep over every job the dispatch cache owes an outcome."""
        return Monitor(self)

    def verdicts(self) -> Verdicts:
        """The receipts-derived outcomes of this workspace's runs, the anti-fabrication read."""
        return Verdicts(self)

    def watch(self, batch_id: str) -> Watch:
        """The live view over an already-dispatched batch, found by id alone.

        Everything it needs is durable (the receipts name the handles, the dispatch cache says
        what became of them), so a process that dispatched nothing can watch another's batch.
        """
        return Watch(self, batch_id, bus=self.receipts(batch_id))
