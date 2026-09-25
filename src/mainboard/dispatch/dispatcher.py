# The CLI-free core of a dispatch: `Dispatcher` submits a job and hands back a `Handle` a caller
# can poll, await, or fetch.

import hashlib
import shlex
from collections.abc import (
    Sequence,  # ruff:ignore[typing-only-standard-library-import]  reason=await_many is inspect.signature()'d in tests, so its Sequence[Handle] annotation must resolve at runtime since=2026-08-17
)
from contextlib import suppress
from pathlib import Path, PurePosixPath
from time import sleep
from typing import TYPE_CHECKING
from uuid import uuid4
from zipfile import BadZipFile

from patos import FrozenModel

from ..context.admission import admit
from ..core.errors import MissionError
from ..core.project import Project
from ..engines.compile.generated import GeneratedFiles
from ..engines.compile.vendor import vendor_root
from ..manifest.loading import load
from ..runtime.job import ToolCall
from . import vocabulary
from .agent import Agent, Scope, SshLink
from .allocation import Allocation
from .collection.collector import Collector
from .jobs import JobSpec
from .mirror import Mirror
from .provenance import Source, SourceTree
from .schedulers import HostUnreachable, failure_reason, pick, read_log, registry
from .shared import HandleId, Watcher, announce, db_file, logger, now, state_path, workspace
from .shipment import Shipment
from .snapshots import CLOSURE, Image, Mirrored, Sealed, Snapshots, writable
from .state.cache import Cache, RunRecord
from .sync import ALWAYS_EXCLUDE, CARD_LEASES, GitignoreFilter, SyncLock, patterns
from .transport import SshTransport
from .vocabulary import JobState, Request, Resources
from .wrapping import connection, wrap

if TYPE_CHECKING:
    from collections.abc import Callable

    from ..context.plan import ExecutionPlan
    from .transport import Machine

# The tool a host runs its own jobs through, so nothing below spells the binary's name.
_TOOL = Project().name

# What never belongs in a vendored path dependency's copy on a host: caches and environments
# beside its sources, which no build reads and one of which is gigabytes. Its own list because
# the vendored tree is its own scope, walked through its links without the workspace's ignores.
_VENDOR_EXCLUDE = ("__pycache__/", "*.pyc", ".git", ".pixi/", ".venv/")


def providing(plan: ExecutionPlan, *, root: str, pinned: str, prefix: str = "") -> ToolCall | None:
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

    root: the host workspace root, where built environments live.
    pinned: the snapshot whose compiled artifact describes the environment.
    prefix: the pinned built environment, its last segment the digest; empty lets the host build
        whatever it reads.
    """
    if plan.containerized:
        return None
    artifact = f"{pinned}/{Project().out_dir}/envs/{plan.env}"
    expect = ("--expect", prefix.rpartition("/")[2]) if prefix else ()
    return ToolCall(args=("provide", plan.env, "--source", artifact, *expect), cwd=root)


# A verdict's process exit code: 0 ok, 1 failed, 2 still running, 3 vanished or unknown.
_VERDICT_EXITS = {"ok": 0, "failed": 1, "running": 2}


class Handle(FrozenModel):
    """A dispatched job, enough to poll, await, or fetch it without re-resolving the host.

    id: the scheduler's handle (PBS or SLURM job id, pueue task id, local run id), always text,
        since pueue's small integers read back as numbers would fail validation inside a poll.
    host: the ssh alias the job runs on, or the declared alias of the provider host rented for it.
    root: the mirror's workspace root on that host, empty for a provider that syncs none. The
        mirror rather than the snapshot, since later log reads, results pulls and post-mortems
        address it and it outlives the snapshot.
    kind: the scheduler (`pbs`/`slurm`/`ssh`/`local`) or provider (`vast`/`hpc-ai`/`modal`) kind
        at submit time, which routes a later probe back to whichever answered for this run.
    fetch_path: the results path recorded at submit time, pulled back by `Dispatcher.fetch`.
    """

    id: HandleId
    host: str
    root: str
    kind: str
    fetch_path: str | None = None


class Verdict(FrozenModel):
    """A terminal outcome of an awaited job, the value `Dispatcher.await_many` yields.

    verdict: `ok`, `failed`, `vanished` or `unknown`.
    exit_code: the process exit status, when the scheduler reported one.
    reason: a one-line cause for a non-ok verdict.
    """

    verdict: str
    exit_code: int | None = None
    reason: str = ""

    @property
    def code(self) -> int:
        """The verdict as a process exit code."""
        return _VERDICT_EXITS.get(self.verdict, 3)

    @property
    def ok(self) -> bool:
        """Whether the job finished cleanly."""
        return self.verdict == "ok"


class Dispatcher:
    """Dispatch a job to a resolved host and hand back a `Handle` to poll, await or fetch.

    Every workspace path a dispatch touches (state database, staged scripts, the mirror's include
    list, pulled results) resolves against the mirror filter's root rather than the cwd, since the
    mirror decides where the workspace begins.
    """

    def __init__(
        self,
        cache: Cache | None = None,
        sync: GitignoreFilter | None = None,
        root: Path | None = None,
    ) -> None:
        """Defaults: the store under the root, a filter rooted there, the root found upward."""
        self.sync = sync or GitignoreFilter(root or workspace())
        self.root = self.sync.root
        self.cache = cache or Cache(db_file(self.root))

    def await_many(
        self, handles: Sequence[Handle], *, interval: float = vocabulary.POLL_SECONDS
    ) -> dict[Handle, Verdict]:
        """Poll every `interval` seconds until each handle is terminal, a `probe` blip retried."""
        verdicts: dict[Handle, Verdict] = {}
        pending = list(handles)
        while pending:
            running: list[Handle] = []
            for handle in pending:
                state = self.probe(handle)
                if state is None or state.verdict == "running":
                    running.append(handle)
                else:
                    verdicts[handle] = self._verdict(handle, state)
            if pending := running:
                sleep(interval)
        return verdicts

    def fetch(self, handle: Handle, *, ssh: SshTransport | None = None) -> None:
        """Collect the handle's results without overwriting conflicting local evidence."""
        if not handle.fetch_path:
            raise LookupError(f"handle {handle.id!r} has no fetch path to pull")
        self.fetch_path(handle.host, root=handle.root, path=handle.fetch_path, ssh=ssh)

    def fetch_path(
        self, host: str, *, root: str, path: str, ssh: SshTransport | None = None
    ) -> int:
        """Collect a file or directory through the host profile's Python, on either OS.

        Source synchronization stays separate from this evidence collection.
        """
        python = load(self.root / Project().manifest).profile(host).python
        try:
            published = Collector(self.root, ssh).pull(
                host, root=root, path=path.rstrip("/"), python=python
            )
        except (ValueError, RuntimeError, BadZipFile) as fault:
            raise MissionError(f"collection of {path} from {host} failed: {fault}") from fault
        logger.info("fetched %s from %s", path, host)
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
            state=vocabulary.HELD,
            verdict=vocabulary.HELD,
            request=asked,
            reason=reason,
        )
        self.cache.record(record)
        logger.warning("%s held for %s: %s", asked.command, asked.target, reason)
        return record

    def local(self, path: str) -> Path:
        """`path` as a file here, a relative one resolved against the workspace root.

        The one place a written-down path becomes a local file, so a command typed in a
        subdirectory reads and writes what it would from the root.
        """
        given = Path(path).expanduser()
        return given if given.is_absolute() else self.root / given

    def pinned(self, root: str, *, source: Source) -> str:
        """The snapshot of the mirror at `root` a job dispatched from `source` runs in.

        Path arithmetic alone, so the job script can be rendered before the connection that
        materialises the snapshot opens; `submit` creates it from this very key, not a later one.
        """
        return Snapshots(root).path(source.key)

    def source(self, command: str = "", *, paths: Sequence[str] = ()) -> Source:
        """Fingerprint the mirror scope and explicit command files without version control."""
        tree = SourceTree(self.root)
        roots = [
            *(paths or [Project().manifest]),
            *(
                (self.root / token).relative_to(self.root).as_posix()
                for token in shlex.split(command)
                if (self.root / token).is_file() and (self.root / token).is_relative_to(self.root)
            ),
        ]
        files = [
            file
            for path in roots
            for file in ([path] if (self.root / path).is_file() else tree.kept(path))
        ]
        return tree.seal(files)[0]

    def stage_listing(self, shipment: Shipment) -> str:
        """Stage `shipment`'s closure listing under the jobs directory, empty for a command.

        Content-addressed by the closure digest, so a wave off one closure stages one file. The
        mirror carries it beside the script, the pin copies exactly the files it names, and the
        job reads the same rows through `MAINBOARD_CLOSURE`.
        """
        if not shipment.sealed:
            return ""
        SourceTree(self.root).archive(shipment.listing)
        return self._stage(shipment.listing_name, shipment.listing.encode())

    def image(
        self, plan: ExecutionPlan, shipment: Shipment, *, listing: str, shipped: Sequence[str]
    ) -> Image:
        """What this dispatch's snapshot copies: the closure listed, or the shipped mirror.

        listing: the staged closure listing, workspace-relative, empty for a command.
        shipped: the allowlist the mirror transfer carried, what a command's snapshot copies.
        """
        if listing:
            return Sealed(listing=listing, needs=shipment.needs, pins=shipment.pins)
        return Mirrored(scope=self.scope(plan, shipped).spec())

    def probe(self, handle: Handle) -> JobState | None:
        """One non-blocking scheduler probe of `handle`, None when the host is unreachable.

        A blip is not a verdict, so a caller polling on its own cadence retries rather than
        recording a state the host never reported.
        """
        try:
            return self.state(handle)
        except HostUnreachable as down:
            logger.warning("%s unreachable, retrying: %s", handle.id, down)
            return None

    def agent(self, plan: ExecutionPlan, *, ssh: SshTransport | None = None) -> Agent:
        """The standard-library agent on `plan.host`, what every mirror and pin talks to.

        ssh: the policy a rental's endpoint rides; a declared host's alias answers for itself.
        """
        policy = ssh or SshTransport()
        return Agent(
            SshLink(plan.host, policy), python=plan.profile.python, patience=policy.deadline
        )

    def scope(
        self, plan: ExecutionPlan, roots: Sequence[str], *, hidden: Sequence[str] = ()
    ) -> Scope:
        """The tree a mirror ships from `roots`, pruned by every rule a host obeys.

        Each repository's file list decides it, then the permanent denylist, the host's excludes
        and the card leases prune it.

        hidden: declared output paths, literal, which only ever travel down.
        """
        excluded = [*ALWAYS_EXCLUDE, *plan.profile.sync.exclude, *CARD_LEASES]
        return self.sync.scope(roots, deny=patterns(excluded, paths=hidden))

    def mirror(
        self,
        plan: ExecutionPlan,
        root: str,
        *,
        ssh: SshTransport | None = None,
        required: Sequence[Sequence[str]] = (),
        extra: Sequence[str] = (),
        fetch: str = "",
    ) -> list[str]:
        """Mirror the workspace to `plan.host`, returning the include paths present locally.

        The workspace and nested `.gitignore` files bound what is sent and deleted, and pruning
        reaches only the include paths, so a host file anywhere else is never touched. Output
        paths declared by this submission or any prior job in the workspace are download-only
        whatever their names or local existence; an explicit resource under one is refused
        before transfer (bind a separate immutable input instead). Ordinary `needs` stay mutable
        mirror links, not immutable snapshots, and `plan.profile.sync.protect` also protects
        unregistered remote artifacts. A path shipped by name that vanishes before the stream
        reaches it fails the whole mirror, so a submission never continues on a partial sync.

        Vendored path dependencies (see `engines.compile.vendor`) ship as their own scope with
        each link replaced by its target: here that tree is links into the source, which a host
        could never follow, so it gets real files its compile leaves alone, and a distribution
        the manifest stopped declaring is pruned without reaching anything beside it.

        ssh: where the transfer lands. A declared host is its own alias in the user's ssh config;
            a machine rented for one job has none, so a policy bound to it names it instead.
        required: groups that must exist locally whole and ship together by name despite the
            allowlist and ignores (a compiled manifest, its lock and the state it was solved from).
        extra: paths outside the allowlist that must still reach the host, such as the staged
            job script, shipped by name the same way.
        """
        policy = ssh or SshTransport()
        scope = plan.profile.sync
        if not scope.include:
            raise LookupError(
                f"nothing to sync to {plan.host!r}; declare [hosts.{plan.host}.sync].include "
                "(or [hosts.defaults.sync].include) before dispatching"
            )
        include = [path for path in scope.include if self.local(path).exists()]
        if stale := [path for path in scope.include if path not in include]:
            logger.warning(
                "skipping %d stale sync include path(s) missing locally: %s",
                len(stale),
                ", ".join(stale),
            )
        if not include:
            raise LookupError(f"every sync include path for {plan.host!r} is missing locally")
        if incomplete := [
            list(group) for group in required if not all(self.local(p).is_file() for p in group)
        ]:
            raise LookupError(
                f"required path group(s) {incomplete} are incomplete; build them before "
                "dispatching"
            )
        named = list(dict.fromkeys((*(path for group in required for path in group), *extra)))
        self.sync.validate_sources((*named, *scope.include))
        with SyncLock(policy.endpoint or plan.host, self.sync.root):
            outputs = self._protected_outputs(fetch, sources=named)
            scopes = [self.scope(plan, include, hidden=outputs)]
            if self.local(vendor_root()).is_dir():
                scopes.append(Scope([vendor_root()], deny=patterns(_VENDOR_EXCLUDE), follow=True))
            Mirror(self.root, self.agent(plan, ssh=policy)).push(
                root,
                scopes=scopes,
                named=named,
                protected=patterns([*CARD_LEASES, *scope.protect], paths=outputs),
            )
        self.cache.mark_synced(plan.host)
        return include

    def run(
        self,
        plan: ExecutionPlan,
        shipment: Shipment,
        *,
        root: str,
        resources: Resources,
        verify: str = "true",
        fetch: str | None = None,
        name: str = "",
        node: str = "",
        gpu_in_select: bool = True,
        sampler: ToolCall | None = None,
        attestation: ToolCall | None = None,
        containerize: Callable[[list[str]], list[str]] | None = None,
        watch: Watcher | None = None,
        prefix: str = "",
        artifact: Sequence[str] = (),
    ) -> Handle:
        """Render `shipment` into a job script for `plan`'s host, submit it, return its handle.

        The script carries a `#PBS` header when the profile's kind calls for one and wraps the
        command in the container runtime when `plan.containerized`. It activates and runs from
        the snapshot this dispatch pins, so a later dispatch of another tree cannot change the
        code under a queued or running job; the handle names the mirror, where the logs and
        results stay once the snapshot is pruned.

        resources: the scheduler request, queue, walltime, memory and GPUs already resolved.
        verify: a preflight command proving the host's activated environment actually runs.
        fetch: a results path recorded on the handle, pulled back by `fetch`.
        node: the ledger slug this run serves, recorded on the run and its receipts.
        gpu_in_select: whether a PBS GPU request belongs in the `select=` chunk.
        sampler: the call the job makes beside its command; the caller decides what a host
            watches about itself.
        attestation: the call the job makes in the foreground before its command, recording
            what the machine looked like as the work started.
        containerize: builds the container argv around `["bash", "-c", cmd]`; required when
            `plan.containerized`.
        watch: announces the host building its environment, the one far-side stage long
            enough to be worth saying.
        prefix: the built environment the job activates, addressed by the content of the
            compiled artifact shipped; empty activates the workspace's own environment.
        artifact: the compiled manifest, lock and state the environment was addressed by,
            shipped so the pinned tree carries the very compile the pin was taken over; a group
            under the generated tree travels only when named.
        """
        container: tuple[str, ...] = ()
        if plan.containerized:
            if containerize is None:
                raise LookupError(
                    f"plan for host {plan.host!r} is containerized but no container argv "
                    "builder was given"
                )
            container = tuple(containerize(["bash", "-c", shipment.command]))
        pinned = self.pinned(root, source=shipment.source)
        listing = self.stage_listing(shipment)
        spec = JobSpec(
            cmd=shipment.command,
            plan=plan,
            root=pinned,
            queue=resources.queue or "",
            walltime=resources.walltime or "",
            select=resources.nodes,
            gpus=resources.gpus,
            account=resources.account,
            mem_gb=resources.mem_gb,
            container=container,
            prefix=prefix,
            pythonpath=":".join(f"{pinned}/{place}".rstrip("/") for place in shipment.imports),
            provide=providing(plan, root=root, pinned=pinned, prefix=prefix),
            sampler=sampler,
            attestation=attestation,
            source=shipment.source.identity,
            commit=shipment.source.commit,
            digest=shipment.source.digest,
            closure=f"{pinned}/{CLOSURE}" if listing else "",
            first_party=":".join(shipment.first_party),
            deferred=":".join(shipment.deferred),
            exports=plan.exports,
        )
        script = self.write_job_script(
            spec, pbs=plan.profile.kind == "pbs", gpu_in_select=gpu_in_select
        )
        handle = self.submit(
            plan,
            root,
            script=script,
            args=(),
            resources=resources,
            verify=verify,
            fetch=fetch,
            name=name,
            node=node,
            shipment=shipment,
            listing=listing,
            containerize=containerize,
            watch=watch,
            prefix=prefix,
            required=[
                *([artifact] if artifact else []),
                # A need may live only on the host, and the sealed snapshot checks it there;
                # only local files need punching through the transfer filters.
                *([need] for need in shipment.needs if self.local(need).is_file()),
                *([pin] for pin in shipment.pins),
            ],
        )
        return Handle(
            id=handle, host=plan.host, root=root, kind=plan.profile.kind, fetch_path=fetch
        )

    def state(self, handle: Handle) -> JobState:
        """One scheduler probe of `handle`, raising `HostUnreachable` when the host is down.

        The unabsorbed `probe`, for a caller that must name the dead host and why (a durable
        sweep reporting it once and moving on) rather than retrying it next tick.
        """
        with connection(handle.host) as remote:
            scheduler = registry.SCHEDULERS.select(handle.kind, default="ssh")
            return scheduler.state(remote, handle.root, handle=handle.id)

    def states(self, handles: Sequence[Handle]) -> dict[str, JobState]:
        """Every handle's state, keyed by id, over one connection to the host they share.

        The batched `state`: a cache months old holds a thousand runs on one box, and asking it
        once rather than a thousand times is what lets a sweep finish. One connection and one
        scheduler answer for all, so every handle must share the first one's host, root and
        kind. A handle the listing misses (a SLURM `squeue` spanning only live jobs, a PBS server
        that purged its history) is asked alone over the same connection, where its real ending
        is. `HostUnreachable` surfaces as from `state`: it is about the host, not any one job.
        """
        if not handles:
            return {}
        shared = handles[0]
        scheduler = registry.SCHEDULERS.select(shared.kind, default="ssh")
        ids = list(dict.fromkeys(handle.id for handle in handles))
        with connection(shared.host) as remote:
            listed = scheduler.states(remote, shared.root, ids)
            return {
                job_id: listed.get(job_id) or scheduler.state(remote, shared.root, handle=job_id)
                for job_id in ids
            }

    def submit(
        self,
        plan: ExecutionPlan,
        root: str,
        *,
        script: str,
        args: Sequence[str],
        resources: Resources,
        required: Sequence[Sequence[str]] = (),
        verify: str = "true",
        fetch: str | None = None,
        name: str = "",
        node: str = "",
        shipment: Shipment | None = None,
        listing: str = "",
        containerize: Callable[[list[str]], list[str]] | None = None,
        watch: Watcher | None = None,
        prefix: str = "",
    ) -> str:
        """Ship the workspace, pin the tree it runs from, dispatch `script`, return the handle.

        Admission runs before any ssh connection, so a request the queue's declared policy
        rejects fails without a round trip, and `verify` turns a broken remote environment into
        a clear diagnosis before the scheduler sees the job. The freshly synced mirror is then
        snapshotted and the scheduler pointed at the snapshot: the mirror stays cheap to sync and
        is what log reads, results pulls and post-mortems address, while the immutable snapshot
        is what the job runs from, so the next dispatch of another tree rewrites the mirror under
        nobody. The run is recorded with its provenance, so a later poll re-derives nothing.

        shipment: what the job runs and ships, its provenance read once so two dispatches of one
            tree share one snapshot; none for a hand-written script, which ships the mirror under
            the workspace's own provenance.
        listing: the staged closure listing a sealed shipment's snapshot copies from, shipped
            beside the script; empty for a command that ships the mirror.
        containerize: builds the container argv around `["bash", "-c", verify]`; required when
            `plan.containerized`, so the preflight runs in the base image a job would.
        watch: announces the host building its environment, a stage invisible from here.
        prefix: the built environment the pinned tree activates, addressed by content.
        """
        admit(
            plan.profile,
            queue=resources.queue or "",
            walltime=resources.walltime or "",
            mem_gb=resources.mem_gb or 0,
        )
        dispatched = shipment or Shipment.of_command(
            script, source=self.source(script, paths=plan.profile.sync.include), imports=()
        )
        dispatched.admit(self.root)
        prepared, staged = self._prepare_script(script)
        with SyncLock(plan.host, self.sync.root), connection(plan.host) as remote:
            shipped = self.mirror(
                plan,
                root,
                required=required,
                extra=[*staged, *([listing] if listing else []), *dispatched.files],
                fetch=fetch or "",
            )
            self._verify(remote, plan, root, verify=verify, containerize=containerize)
            pinned = Snapshots(root).pin(
                self.agent(plan),
                key=dispatched.source.key,
                image=self.image(plan, dispatched, listing=listing, shipped=shipped),
                results=fetch or "",
                prefix=prefix,
                environment=plan.env,
                commit=dispatched.source.commit,
                digest=dispatched.source.digest,
                script=prepared if staged else "",
            )
            self._prime(remote, plan, pinned, root, watch, prefix=prefix)
            try:
                handle = pick(plan.profile).submit(
                    remote,
                    pinned,
                    script=Snapshots.script(prepared) if staged else prepared,
                    args=args,
                    resources=resources,
                )
            except SystemExit as error:
                raise SystemExit(f"submission to host {plan.host!r} failed: {error}") from None
            self.cache.record(
                RunRecord(
                    handle=handle,
                    target=plan.host,
                    kind=plan.profile.kind,
                    script=dispatched.spelling if shipment is not None else prepared,
                    args=" ".join(shlex.quote(a) for a in args),
                    submitted_at=now(),
                    fetch_path=fetch,
                    name=name,
                    node=node,
                    source=dispatched.source.key,
                    commit=dispatched.source.commit,
                    digest=dispatched.source.digest,
                )
            )
        logger.info("%s -> %s on %s (%s)", prepared, handle, plan.host, dispatched.source.identity)
        return handle

    def allocating(
        self,
        plan: ExecutionPlan,
        shipment: Shipment,
        resources: Resources,
        *,
        name: str = "",
        node: str = "",
        evidence: str,
    ) -> Allocation:
        """Reserve the existing job row before any provider creation can be attempted."""
        shipment.admit(self.root)
        label = f"mainboard-{uuid4().hex}"
        request = Request(
            target=plan.host,
            command=shipment.spelling,
            name=name,
            node=node,
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
        )
        record = RunRecord(
            handle=label,
            target=plan.host,
            kind=plan.profile.kind,
            script=shipment.spelling,
            args="",
            submitted_at=now(),
            fetch_path=shipment.fetch or None,
            name=name,
            node=node,
            source=shipment.source.key,
            commit=shipment.source.commit,
            digest=shipment.source.digest,
            state=vocabulary.PREPARED,
            verdict=vocabulary.PREPARED,
            evidence=evidence,
            creation=label,
            request=request,
        )
        try:
            self.cache.reserve(record)
        except ValueError as unresolved:
            raise MissionError(str(unresolved)) from unresolved
        return Allocation(cache=self.cache, record=record)

    def write_job_script(self, spec: JobSpec, *, pbs: bool, gpu_in_select: bool = True) -> str:
        """Render `spec` under the jobs directory and answer its workspace-relative path.

        Content-addressed, so repeated runs reuse the file rather than growing the directory.
        Workspace-relative because the mirror recreates it on the host and the scheduler runs it
        there.
        """
        text = spec.render(pbs=pbs, gpu_in_select=gpu_in_select).encode()
        return self._stage(f"job-{hashlib.sha256(text).hexdigest()}.sh", text)

    def _prepare_script(self, script: str) -> tuple[str, tuple[str, ...]]:
        """Stage a local script, answering its host-safe path and its sync source.

        A bare name stays unchanged (a future on-host executor resolves it in-repository). An
        explicit path must exist locally: forwarding an unresolved one would fail on the host
        later with no way to guarantee what it runs.
        """
        try:
            content = self.local(script).read_bytes()
        except FileNotFoundError as error:
            if Path(script).name == script:
                return script, ()
            raise FileNotFoundError(
                f"cannot submit script {script!r}: the local file does not exist, so it "
                "cannot be shipped to the host"
            ) from error
        staged = self._stage(f"job-{hashlib.sha256(content).hexdigest()}.sh", content)
        return staged, (staged,)

    def _protected_outputs(self, fetch: str, *, sources: Sequence[str] = ()) -> list[str]:
        """Download-only output paths, refusing any that overlaps explicitly shipped input.

        The workspace cache spans host aliases and keeps completed evidence. Paths are literal,
        so glob characters in a real filename never widen what they protect.
        """
        recorded = (run.fetch_path for run in self.cache.recent(limit=None))
        outputs = []
        for path in sorted({path for path in (fetch, *recorded) if path}):
            relative = writable(path)
            if not relative or relative == ".":
                raise ValueError(
                    f"declared output must be a relative path below the workspace: {path!r}"
                )
            if any(
                PurePosixPath(source).is_relative_to(relative)
                or PurePosixPath(relative).is_relative_to(source)
                for source in sources
            ):
                raise ValueError(
                    f"explicit input overlaps declared output {relative!r}; "
                    "bind the selected data under a separate immutable input path"
                )
            outputs.append(relative)
        return outputs

    def _stage(self, name: str, content: bytes) -> str:
        """Atomically stage exact bytes and answer their workspace-relative path."""
        path = state_path(self.root) / "jobs" / name
        with GeneratedFiles(directory=path.parent).locked() as files:
            files.write(path, content)
        return path.relative_to(self.root).as_posix()

    def _verdict(self, handle: Handle, state: JobState) -> Verdict:
        """Persist a terminal state and project it onto a `Verdict`.

        Only a job not ending `ok` has its log read for a reason, so a clean run pays no second
        round trip.
        """
        with suppress(LookupError):
            run = self.cache.run(handle.id, target=handle.host)
            self.cache.resolve(run, state.state, state.exit_code, state.verdict)
        if state.verdict == "ok":
            return Verdict(verdict="ok", exit_code=state.exit_code)
        with connection(handle.host) as remote:
            log = read_log(remote, handle.root, handle=handle.id)
        return Verdict(
            verdict=state.verdict,
            exit_code=state.exit_code,
            reason=failure_reason(log, state.exit_code),
        )

    def _prime(
        self,
        remote: Machine,
        plan: ExecutionPlan,
        pinned: str,
        root: str,
        watch: Watcher | None = None,
        *,
        prefix: str = "",
    ) -> None:
        """Build the environment `pinned` activates before submitting, refusing a failed build.

        watch: receives the successful build's announcement.
        """
        call = providing(plan, root=root, pinned=pinned, prefix=prefix)
        if call is None:
            return
        command = f"{shlex.join([_TOOL, *call.args])} >/dev/null"
        retcode, _, err = remote["bash"][
            ["-lc", wrap(plan, root, command=command, activate=False)]
        ].run(retcode=None)
        if retcode:
            raise SystemExit(
                f"could not build {plan.env} on {plan.host}: {failure_reason(str(err))}"
            )
        told = f"built {plan.env} on {plan.host} for {pinned}"
        logger.info("%s", told)
        (watch or announce)(told)

    def _verify(
        self,
        remote: Machine,
        plan: ExecutionPlan,
        root: str,
        *,
        verify: str,
        containerize: Callable[[list[str]], list[str]] | None,
    ) -> None:
        """Fail fast, in one plain sentence, when `plan.host` cannot run the job it is sent.

        `verify` runs through the activation wrap every job depends on, so a broken environment
        (a stale install, a dependency the sync never shipped) is diagnosed before the scheduler
        sees the job rather than buried as a traceback in its log. Then the host's tool is asked
        for the verb a job script hands over to, found the way the script finds it: a tool from
        before jobs ran through it would queue the job and end it at start with an unknown
        command and no exit artifact.
        """
        body = wrap(plan, root, command=verify, containerize=containerize)
        retcode, _, err = remote["bash"][["-lc", body]].run(retcode=None)
        if retcode != 0:
            raise SystemExit(f"environment on {plan.host!r} is broken: {failure_reason(err)}")
        # An unknown verb's `--help` prints the root help and succeeds, so the usage line the
        # verb's own help opens with is what tells the two tools apart.
        usage = shlex.quote(f"Usage: {_TOOL} job ")
        runs = wrap(plan, root, command=f"{_TOOL} job --help | grep -q {usage}", activate=False)
        retcode, _, err = remote["bash"][["-lc", runs]].run(retcode=None)
        if retcode != 0:
            raise SystemExit(
                f"{_TOOL} on {plan.host!r} cannot run a job ({failure_reason(err)}); run "
                f"`{_TOOL} setup {plan.host}` to install this version there"
            )
