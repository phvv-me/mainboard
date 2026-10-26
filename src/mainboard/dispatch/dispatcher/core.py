# The CLI-free core of a dispatch: `Dispatcher` submits a job and hands back a `Handle` a caller
# can poll, await, or fetch.

import hashlib
import re
import shlex
from typing import TYPE_CHECKING
from uuid import uuid4
from zipfile import BadZipFile

from ...context.admission import admit
from ...context.plan import environment_prefix
from ...core.errors import MissionError
from ...core.project import Project
from ...engines.compile.pixi.lockfile import Lockfile
from ...log import logger
from ...runtime.job import ToolCall
from ...state.storage.holds import holding
from ..agent import program
from ..collection.collector import Collector
from ..hosts.allocation import Allocation
from ..jobs import JobSpec, vocabulary
from ..jobs.vocabulary import Request
from ..mirroring.snapshots import Snapshots
from ..mirroring.sync import SyncLock
from ..schedulers import Held, Pbs, failure_reason, kind_of, pick
from ..shared import announce, now, state_dir
from ..state.cache import RunRecord
from ..transport.core import SshTransport
from ..transport.shells import Dialect
from ..transport.targets import probe_capabilities, resolve
from ..wrapping import connection, guarded, wrap
from .order import Handle, Order
from .shipping import Shipper, pin_key

if TYPE_CHECKING:
    from ..schedulers import Scheduler
    from ..transport.core import Machine

# The tool a host runs its own jobs through, so nothing below spells the binary's name: the
# package's own script, which every release a host may still run installs.
_TOOL = Project().package


def _providing(order: Order, *, pinned: str) -> ToolCall | None:
    """The idempotent call building the environment `pinned` activates, None for a container.

    The dispatch runs it after pinning so a wave finds its environment built, and the job's
    runner runs it again on the node so a pruned or unfinished prefix is built rather than used
    half-installed; every repeat costs a stat. Only the call runs from the mirror, where a host
    keeps built environments (a relative path in the command still names the snapshot's frozen
    code), and it builds from the snapshot's hardlinked artifact, so what a job activates and
    what was built are the same content. `--expect` carries the pinned digest, so a host that
    reads the artifact differently names both addresses and both pixis instead of building a
    stray environment beside the one the wave waits for and reporting success. A containerized
    plan activates its image, which no lock on this host describes.

    pinned: the snapshot whose compiled artifact describes the environment.
    """
    plan = order.plan
    if plan.containerized:
        return None
    artifact = f"{pinned}/{Project().out_dir()}/envs/{plan.env}"
    expect = ("--expect", order.prefix.rpartition("/")[2]) if order.prefix else ()
    return ToolCall(args=("provide", plan.env, "--source", artifact, *expect), cwd=order.root)


class Dispatcher(Shipper):
    """Dispatch a job to a resolved host and hand back a `Handle` to poll, await or fetch."""

    @staticmethod
    def _checkpoint_dir(root: str, *, name: str) -> str:
        """Where every attempt of the run named `name` keeps what it resumes from, on its host.

        Under the mirror's state directory rather than the pinned tree, which is immutable and
        replaced when the code changes; a resumed run usually runs fixed code.
        """
        slug = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-") or "run"
        return f"{root}/{state_dir()}/checkpoints/{slug}"

    def _handed(
        self, order: Order, scheduler: Scheduler, remote: Machine, *, pinned: str, script: str
    ) -> str:
        """Give the job to `scheduler`, answer its handle, and leave the run recorded.

        A held line's job is recorded before its entry is written, as a creation intent, so work
        queued for a node is never outside the registry: a refusal closes the intent, while a
        reply lost in transit leaves it `submitting` for reconciliation by its label instead of
        resubmitting.
        """
        plan, shipment = order.plan, order.shipment
        record = RunRecord(
            handle="",
            target=plan.host,
            kind=kind_of(plan.profile, order.resources.queue or ""),
            script=shipment.spelling,
            args="",
            submitted_at=now(),
            fetch_path=shipment.fetch or None,
            name=order.name,
            node=order.node,
            source=pin_key(shipment.source, order.prefix),
            commit=shipment.source.commit,
            digest=shipment.source.digest,
            project=self.project,
        )
        if not isinstance(scheduler, Held):
            handle = scheduler.submit(remote, pinned, script=script, resources=order.resources)
            self.cache.record(record.model_copy(update={"handle": handle}))
            return handle
        label = f"mainboard-{uuid4().hex}"
        pending = record.model_copy(
            update={
                "handle": label,
                "creation": label,
                "state": vocabulary.Word.PREPARED,
                "verdict": vocabulary.Word.PREPARED,
            }
        )
        self.cache.record(pending)
        intent = Allocation(cache=self.cache, record=pending)
        intent.begin()
        try:
            handle = scheduler.enqueue(
                remote, pinned, script=script, resources=order.resources, label=label
            )
        except MissionError:
            intent.refused()
            intent.interrupted()
            raise
        return intent.created(handle)

    def _pin(self, order: Order, remote: Machine, *, script: str, listing: str) -> str:
        """Mirror the workspace, prove the host can run the job, and pin the tree it runs from.

        The mirror stays cheap to sync and is what log reads, results pulls and post-mortems
        address, while the immutable snapshot is what the job runs from, so the next dispatch of
        another tree rewrites the mirror under nobody.

        script: the staged job script. listing: the staged closure listing, empty for a command.
        """
        plan, shipment = order.plan, order.shipment
        required = [
            *([order.artifact] if order.artifact else []),
            # A need may live only on the host, and the sealed snapshot checks it there; only
            # local files need punching through the transfer filters.
            *([need] for need in shipment.needs if self.local(need).is_file()),
            *([pin] for pin in shipment.pins),
            *shipment.carried(script=script, listing=listing),
        ]
        shipped = self.mirror(plan, order.root, required=required, fetch=shipment.fetch)
        self._verify(remote, order)
        return Snapshots(order.root).pin(
            self.agent(plan),
            key=pin_key(shipment.source, order.prefix),
            image=self.image(plan, shipment, listing=listing, shipped=shipped),
            results=shipment.fetch,
            prefix=order.prefix,
            environment=plan.env,
            commit=shipment.source.commit,
            digest=shipment.source.digest,
            script=script,
        )

    def _prime(self, remote: Machine, order: Order, pinned: str) -> None:
        """Build the environment `pinned` activates before submitting, refusing a failed build.

        The build is announced through the order's watcher, the one far-side stage long enough
        to be worth saying.
        """
        plan = order.plan
        call = _providing(order, pinned=pinned)
        if call is None:
            return
        command = f"{shlex.join([_TOOL, *call.args])} >/dev/null"
        retcode, _, err = remote["bash"][
            ["-lc", guarded(wrap(plan, order.root, command=command, activate=False), plan)]
        ].run(retcode=None)
        if retcode:
            raise SystemExit(f"could not build {plan.env} on {plan.host}: {failure_reason(err)}")
        told = f"built {plan.env} on {plan.host} for {pinned}"
        logger.info(told)
        (order.watch or announce)(told)

    def _spec(self, order: Order, *, pinned: str, listing: str) -> JobSpec:
        """The job `order` runs from the snapshot `pinned`, wrapped in its container if any."""
        plan, shipment, resources = order.plan, order.shipment, order.resources
        if plan.containerized and order.containerize is None:
            raise LookupError(
                f"plan for host {plan.host!r} is containerized but no container argv builder "
                "was given"
            )
        return JobSpec(
            cmd=shipment.command,
            plan=plan,
            root=pinned,
            queue=resources.queue or "",
            walltime=resources.walltime or "",
            select=resources.nodes,
            gpus=resources.gpus,
            account=resources.account,
            mem_gb=resources.mem_gb,
            container=(
                tuple(order.containerize(["bash", "-c", shipment.command]))
                if order.containerize and plan.containerized
                else ()
            ),
            prefix=order.prefix,
            pythonpath=":".join(f"{pinned}/{place}".rstrip("/") for place in shipment.imports),
            provide=_providing(order, pinned=pinned),
            sampler=order.sampler,
            attestation=order.attestation,
            source=shipment.source.identity,
            commit=shipment.source.commit,
            digest=shipment.source.digest,
            closure=program.closure_path(pinned) if listing else "",
            first_party=":".join(shipment.first_party),
            deferred=":".join(shipment.deferred),
            checkpoint=Dispatcher._checkpoint_dir(order.root, name=order.name)
            if order.name
            else "",
            attempt=resources.attempt,
            exports=plan.exports,
        )

    def _submit(self, order: Order, *, script: str, listing: str) -> str:
        """Ship the workspace, pin the tree the job runs from, hand it over, answer the handle.

        Admission runs before any ssh connection, so a request the queue's declared policy
        rejects fails without a round trip. The run is recorded with its provenance, so a later
        poll re-derives nothing.
        """
        plan, resources = order.plan, order.resources
        admit(
            plan.profile,
            queue=resources.queue or "",
            walltime=resources.walltime or "",
            mem_gb=resources.mem_gb or 0,
        )
        order.shipment.admit(self.root)
        with SyncLock(plan.host, self.sync.root), connection(plan.host) as remote:
            pinned = self._pin(order, remote, script=script, listing=listing)
            scheduler = pick(plan.profile, resources.queue or "")
            # A held line's runner provisions its environment inside the allocation; nothing is
            # built on the login node for it.
            if not isinstance(scheduler, Held):
                self._prime(remote, order, pinned)
            try:
                handle = self._handed(
                    order, scheduler, remote, pinned=pinned, script=Snapshots.script(script)
                )
            except SystemExit as error:
                raise SystemExit(f"submission to host {plan.host!r} failed: {error}") from None
        logger.info(
            "{} -> {} on {} ({})", script, handle, plan.host, order.shipment.source.identity
        )
        return handle

    def _verify(self, remote: Machine, order: Order) -> None:
        """Fail fast, in one plain sentence, when the host cannot run the job it is sent.

        A no-op runs through the activation wrap every job depends on, so a broken environment
        (a stale install, a dependency the sync never shipped) is diagnosed before the scheduler
        sees the job rather than buried as a traceback in its log; a containerized plan runs it
        in the base image a job would. Then the host's tool is asked for the verb a job script
        hands over to, found the way the script finds it: a tool from before jobs ran through it
        would queue the job and end it at start with an unknown command and no exit artifact.
        """
        plan, root = order.plan, order.root
        body = wrap(plan, root, command="true", containerize=order.containerize)
        retcode, _, err = remote["bash"][["-lc", body]].run(retcode=None)
        if retcode != 0:
            raise SystemExit(f"environment on {plan.host!r} is broken: {failure_reason(err)}")
        # An unknown verb's `--help` prints the root help and succeeds, so the usage line the
        # verb's own help opens with is what tells the two tools apart, under whichever name the
        # host's release titles its help.
        usage = shlex.quote(f"Usage: ({'|'.join(Project().names)}) execute ")
        check = f"{_TOOL} execute --help | grep -qE {usage}"
        runs = wrap(plan, root, command=check, activate=False)
        retcode, _, err = remote["bash"][["-lc", runs]].run(retcode=None)
        if retcode != 0:
            raise SystemExit(
                f"{_TOOL} on {plan.host!r} cannot run a job ({failure_reason(err)}); run "
                f"`{Project().name} host setup {plan.host}` to install this version there"
            )

    def allocating(self, order: Order, *, evidence: str) -> Allocation:
        """Reserve the existing job row before any provider creation can be attempted."""
        plan, shipment, resources = order.plan, order.shipment, order.resources
        shipment.admit(self.root)
        label = f"mainboard-{uuid4().hex}"
        request = Request(
            target=plan.host,
            command=shipment.spelling,
            name=order.name,
            node=order.node,
            fetch=shipment.fetch or None,
            env=plan.env,
            container=resources.container,
            queue=resources.queue or "",
            walltime=resources.walltime or "",
            mem_gb=resources.mem_gb or 0,
            gpus=resources.gpus,
            gpu_name=resources.gpu_name,
            max_usd=resources.max_usd,
            nodes=resources.nodes,
            spot=resources.spot,
            arch=resources.arch,
        )
        record = RunRecord(
            handle=label,
            target=plan.host,
            kind=plan.profile.kind,
            script=shipment.spelling,
            args="",
            submitted_at=now(),
            fetch_path=shipment.fetch or None,
            name=order.name,
            node=order.node,
            source=shipment.source.key,
            commit=shipment.source.commit,
            digest=shipment.source.digest,
            state=vocabulary.Word.PREPARED,
            verdict=vocabulary.Word.PREPARED,
            evidence=evidence,
            creation=label,
            request=request,
            project=self.project,
        )
        try:
            self.cache.creations.reserve(record)
        except ValueError as unresolved:
            raise MissionError(str(unresolved)) from unresolved
        return Allocation(cache=self.cache, record=record)

    def fetch_path(
        self, host: str, *, root: str, path: str, ssh: SshTransport | None = None
    ) -> int:
        """Collect a file or directory through the host's workspace environment, on either OS.

        Source synchronization stays separate from this evidence collection.
        """
        profile = holding(Project().manifest(self.root)).profile(host)
        if not profile.platform:
            try:
                facts = self.cache.hosts.recorded(host).capabilities
            except LookupError:
                facts = None
            profile = resolve(profile, facts or probe_capabilities(host, ssh=ssh))
        python = Dialect().python(
            environment_prefix(root, env=profile.env),
            host=host,
            python=Lockfile(self.root).python_request(profile.platform),
        )
        try:
            published = Collector(self.root, ssh).pull(
                host, root=root, path=path.rstrip("/"), python=python
            )
        except (ValueError, RuntimeError, BadZipFile) as fault:
            raise MissionError(f"collection of {path} from {host} failed: {fault}") from fault
        logger.info(
            "fetched {} from {} into the lake; `lake materialize {}` writes it to the tree",
            path,
            host,
            path,
        )
        return published

    def hold(self, asked: Request, *, reason: str) -> RunRecord:
        """Keep a dispatch a target's quota refused, so a later sweep can ask for it again.

        The row lives in the registry every run lives in, which the durable sweep, `jobs`,
        `watch` and `verdict` read: a request held only in the dispatching process vanishes with
        it, which is how four jobs of a thirteen-job wave went missing until someone counted the
        logs hours later (miyabi-g, njobs-g quota, 2026-09-04). No scheduler took the job, so its
        `held-` handle is ours, derived from the request: holding it twice keeps one row, which
        is dropped once the request goes through and the real handle takes its place.

        reason: what the target said when it refused, kept as the row's detail.
        """
        seed = f"{asked.target}\n{asked.name}\n{asked.command}"
        handle = f"held-{hashlib.blake2s(seed.encode(), digest_size=5).hexdigest()}"
        try:
            stamped = self.cache.run(handle, asked.target).submitted_at
        except LookupError:
            stamped = now()
        record = RunRecord(
            handle=handle,
            target=asked.target,
            kind="",
            script=asked.command,
            args="",
            submitted_at=stamped,
            fetch_path=asked.fetch,
            name=asked.name,
            node=asked.node,
            state=vocabulary.Word.HELD,
            verdict=vocabulary.Word.HELD,
            request=asked,
            reason=reason,
            project=self.project,
        )
        self.cache.record(record)
        logger.warning("{} held for {}: {}", asked.command, asked.target, reason)
        return record

    def run(self, order: Order) -> Handle:
        """Render `order` into a job script for its host, submit it, return its handle.

        The script carries a `#PBS` header when the profile's kind calls for one and wraps the
        command in the container runtime when the plan is containerized. It activates and runs
        from the snapshot this dispatch pins, so a later dispatch of another tree cannot change
        the code under a queued or running job; the handle names the mirror, where the logs and
        results stay once the snapshot is pruned.
        """
        plan, shipment, queue = order.plan, order.shipment, order.resources.queue or ""
        listing = self.stage_listing(shipment)
        pinned = self.pinned(order.root, source=shipment.source, prefix=order.prefix)
        script = self.write_job_script(
            self._spec(order, pinned=pinned, listing=listing),
            pbs=isinstance(pick(plan.profile, queue), Pbs),
        )
        return Handle(
            id=self._submit(order, script=script, listing=listing),
            host=plan.host,
            root=order.root,
            kind=kind_of(plan.profile, queue),
            fetch_path=shipment.fetch or None,
        )
