# The CLI-free core of a dispatch: submit a job and get a handle back. `Dispatcher` holds the
# reusable core every dispatch shares, and hands back a `Handle` a caller can poll, await, or
# fetch.

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

# What never belongs in a vendored path dependency's copy on a host: the caches and the
# environments a package accumulates beside its sources, none of which a build reads and one
# of which is gigabytes. Its own list because the vendored tree is its own scope, walked through
# its links and without the workspace's ignore files.
_VENDOR_EXCLUDE = ("__pycache__/", "*.pyc", ".git", ".pixi/", ".venv/")


def providing(plan: ExecutionPlan, *, root: str, pinned: str, prefix: str = "") -> str:
    """The line that builds the immutable environment `pinned` activates, empty when it has none.

    One spelling for the two places that need it: the dispatch runs it after pinning, so a wave
    finds its environment already built, and the job runs it again on the node, so a job whose
    prefix was pruned or never finished builds it rather than dying in a half-installed one. It
    is idempotent by construction, so the second call through the ninth cost a stat.

    It runs from the mirror, because that is where a host keeps its built environments, while
    the artifact it builds from is the snapshot's own hardlinked copy: the description a job
    activates and the environment it gets are then the same content by construction. It runs in
    a subshell, so the directory it changes into never becomes the job's own: a relative path in
    the command would otherwise name the mirror's mutable code under the snapshot's provenance.

    The digest the dispatch pinned rides along, so the host builds the environment this job will
    actually activate or says which two addresses it reached and which two pixis reached them.
    Without it a host that read the shipped artifact differently built a whole environment
    beside the one every job of the wave was waiting for and reported success.

    A containerized plan has no such environment: what it activates is the image, which no lock
    on this host describes.

    plan: the resolved execution context whose environment is being built.
    root: the workspace root on the host, where the built environments live.
    pinned: the snapshot whose compiled artifact describes the environment.
    prefix: the built environment this dispatch pinned, whose last segment is that digest; empty
        leaves the host to build whatever it reads.
    """
    if plan.containerized:
        return ""
    artifact = f"{pinned}/{Project().out_dir}/envs/{plan.env}"
    expect = f" --expect {shlex.quote(prefix.rpartition('/')[2])}" if prefix else ""
    build = (
        f"{_TOOL} provide {shlex.quote(plan.env)} "
        f"--source {shlex.quote(artifact)}{expect} >/dev/null"
    )
    return f"( cd {shlex.quote(root)} && {build} )"


# How a finished verdict maps to a process exit code: 0 ok, 1 failed, 2 still running, 3
# vanished/unknown. A caller can branch on this without re-deriving it.
_VERDICT_EXITS = {"ok": 0, "failed": 1, "running": 2}


class Handle(FrozenModel):
    """A dispatched job, enough to poll, await, or fetch it without re-resolving the host.

    id: the scheduler's job handle (PBS job id, pueue task id, SLURM job id, the local run id),
        always text: pueue hands out small integers and a caller who read one back as a number
        would otherwise fail validation deep inside a status poll.
    host: the ssh alias the job runs on, or the declared alias of the provider host it was
        rented for.
    root: the mirror's workspace root on that host, empty for a provider that syncs no
        workspace. The mirror rather than the snapshot the job runs from, since this is what a
        later log read, results pull and post-mortem address, and it outlives the snapshot.
    kind: the kind used at submit time, a scheduler's (`pbs` / `slurm` / `ssh` / `local`) or a
        provider's (`vast` / `hpc-ai` / `modal`), which is what routes a later probe back to
        whichever of the two answered for this run.
    fetch_path: the results path recorded at submit time, pulled back by `Dispatcher.fetch`.
    """

    id: HandleId
    host: str
    root: str
    kind: str
    fetch_path: str | None = None


class Verdict(FrozenModel):
    """A terminal outcome of an awaited job, the value `Dispatcher.await_many` yields.

    verdict: one word, `ok` / `failed` / `vanished` / `unknown` (terminal forms only).
    exit_code: the process exit status, when the scheduler reported one.
    reason: a one-line cause for a non-ok verdict, else "".
    """

    verdict: str
    exit_code: int | None = None
    reason: str = ""

    @property
    def code(self) -> int:
        """The verdict as a process exit code (0 ok, 1 failed, 3 vanished/unknown)."""
        return _VERDICT_EXITS.get(self.verdict, 3)

    @property
    def ok(self) -> bool:
        """Whether the job finished cleanly."""
        return self.verdict == "ok"


def held_handle(asked: Request) -> str:
    """The local id a held dispatch is recorded under, derived from the request itself.

    A scheduler handle names a job a scheduler took, and nothing took this one, so the id is
    ours and says so. Deriving it from the request is what makes holding the same job twice one
    row instead of two, and what lets a later sweep replace it with the real handle.

    asked: the dispatch being held.
    """
    seed = f"{asked.target}\n{asked.name}\n{asked.command}"
    return f"held-{hashlib.blake2s(seed.encode(), digest_size=5).hexdigest()}"


class Dispatcher:
    """Dispatch a job to a resolved host and hand back a `Handle` to poll/await/fetch.

    Every workspace path a dispatch touches, the state database, the staged job scripts, the
    mirror's own include list and a pulled results path, resolves against one root rather than
    against the directory the command was typed in. That root is the mirror filter's, since the
    mirror is what decides where the workspace begins.
    """

    def __init__(
        self,
        cache: Cache | None = None,
        sync: GitignoreFilter | None = None,
        root: Path | None = None,
    ) -> None:
        """cache: the dispatch state store, the one under `root` when None.

        sync: the mirror's ignore filter, one rooted at `root` when None.
        root: the workspace root, discovered upward from the cwd when None.
        """
        self.sync = sync or GitignoreFilter(root or workspace())
        self.root = self.sync.root
        self.cache = cache or Cache(db_file(self.root))

    def await_many(
        self, handles: Sequence[Handle], *, interval: float = vocabulary.POLL_SECONDS
    ) -> dict[Handle, Verdict]:
        """Block until every handle is terminal, returning each one's `Verdict`.

        Polls the scheduler for the still-running handles each `interval` seconds. A transient
        `HostUnreachable` on one tick is not a verdict, so that handle is simply retried on the
        next tick rather than failing the wait.
        """
        verdicts: dict[Handle, Verdict] = {}
        pending = list(handles)
        while pending:
            still_running: list[Handle] = []
            for handle in pending:
                resolved = self.probe(handle)
                if resolved is None or resolved.verdict == "running":
                    still_running.append(handle)
                    continue
                verdicts[handle] = self._verdict(handle, resolved)
            pending = still_running
            if pending:
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
        """Collect a file or directory through remote Python, on either OS.

        The host profile's python command bootstraps standard-library filesystem operations.
        Source synchronization remains separate from evidence collection.
        """
        profile = load(self.root / Project().manifest).profile(host)
        try:
            published = Collector(self.root, ssh).pull(
                host,
                root=root,
                path=path.rstrip("/"),
                python=profile.python,
            )
        except (ValueError, RuntimeError, BadZipFile) as fault:
            raise MissionError(f"collection of {path} from {host} failed: {fault}") from fault
        logger.info("fetched %s from %s", path, host)
        return published

    def hold(self, asked: Request, *, reason: str) -> RunRecord:
        """Keep a dispatch a target's quota refused, so a later sweep can ask for it again.

        The row goes into the same registry every dispatched run lives in, because that registry
        is what the durable sweep reads and what `jobs`, `watch` and `verdict` answer from: a
        request held only in the dispatching process is a job that disappears the moment that
        process ends, which is exactly how four jobs of a thirteen job wave went missing until
        someone counted the logs hours later (miyabi-g, njobs-g quota, 2026-09-04).

        Its handle is this workstation's own, not a scheduler's, since no scheduler ever took the
        job. It is derived from the request, so holding the same job twice keeps one row rather
        than growing one per attempt, and the row is dropped outright once the request goes
        through and the real handle takes its place.

        asked: the dispatch to make again when the quota has room.
        reason: what the target said when it refused, kept as the row's own detail.
        """
        handle = held_handle(asked)
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
        """`path` as a real file here: a workspace-relative name resolved against the root.

        The one place a dispatch turns a written-down path into a file on this machine, so a
        command typed in a subdirectory reads and writes the same files it would from the
        workspace root. An absolute path is already a location and passes through.
        """
        given = Path(path).expanduser()
        return given if given.is_absolute() else self.root / given

    def pinned(self, root: str, *, source: Source) -> str:
        """Where on `root`'s host a job dispatched from `source` runs: that tree's snapshot.

        Path arithmetic alone, so a dispatch can render the job script that activates and runs
        from this directory before it opens the connection that materialises it. `submit` is
        what creates it, from the very key this reading carries rather than from a second one
        taken later.

        root: the workspace root on the host, the mirror the snapshot is taken from.
        source: the dispatching tree, read once by `source`.
        """
        return Snapshots(root).path(source.key)

    def source(self, command: str = "", *, paths: Sequence[str] = ()) -> Source:
        """Fingerprint the mirror scope and explicit command files without version control."""
        tree = SourceTree(self.root)
        roots = list(paths or [Project().manifest])
        roots.extend(
            (self.root / token).relative_to(self.root).as_posix()
            for token in shlex.split(command)
            if (self.root / token).is_file() and (self.root / token).is_relative_to(self.root)
        )
        files = [
            file
            for path in roots
            for file in ([path] if (self.root / path).is_file() else tree.kept(path))
        ]
        source, _ = tree.seal(files)
        return source

    def stage_listing(self, shipment: Shipment) -> str:
        """Write `shipment`'s closure listing under the jobs directory, empty for a command.

        Content-addressed by the closure digest like the job script, so a wave off one closure
        stages one file. The mirror carries it beside the script, the pin copies exactly the
        files it names, and the job reads the same rows through `MAINBOARD_CLOSURE`.
        """
        if not shipment.sealed:
            return ""
        SourceTree(self.root).archive(shipment.listing)
        return self._stage(shipment.listing_name, shipment.listing.encode())

    def image(
        self, plan: ExecutionPlan, shipment: Shipment, *, listing: str, shipped: Sequence[str]
    ) -> Image:
        """What the snapshot of this dispatch copies: the closure listed, or the shipped mirror.

        plan: the resolved execution context, whose profile names what the mirror excludes.
        shipment: what the dispatch runs and ships.
        listing: the staged closure listing, workspace-relative, empty for a command.
        shipped: the allowlist the mirror transfer carried, what a command's snapshot copies.
        """
        if listing:
            return Sealed(listing=listing, needs=shipment.needs, pins=shipment.pins)
        return Mirrored(scope=self.scope(plan, shipped).spec())

    def probe(self, handle: Handle) -> JobState | None:
        """One non-blocking scheduler probe of `handle`, the read a status view wants.

        Returns None on a transient blip (an unreachable host on this tick) rather than a
        verdict, so a caller polling on its own cadence retries instead of recording a state
        the host never actually reported. `await_many` is this same probe under a wait loop.
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
        """The workspace tree a mirror ships from `roots`, pruned by every rule a host obeys.

        Each repository's own file list decides it, then the permanent denylist, the host's own
        excludes and the card leases prune it, and `hidden` names literal paths that only ever
        travel down.

        plan: the resolved execution context, whose profile names what the host excludes.
        roots: the include paths the tree starts from.
        hidden: declared output paths, never uploaded.
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
        """Mirror the workspace to `plan.host`; git-ignored files and the denylist skipped.

        The workspace and nested `.gitignore` files are the primary send and delete boundary,
        and pruning reaches only the include paths: a file the host holds anywhere else is
        never touched. Declared output paths from this submission and prior jobs in this
        workspace are always download-only, regardless of their names or local existence. An
        explicit resource underneath an output path is refused before transfer; bind a separate
        immutable input instead. Ordinary `needs` remain mutable mirror links, not immutable
        input snapshots. `plan.profile.sync.protect` additionally protects unregistered remote
        artifacts. `required` names groups of paths that must ship together despite being
        outside the allowlist or git-ignored (a compiled manifest with its lock and the state
        naming what that lock was solved from, say): each group is required to exist locally as
        a whole, and ships by name whatever the rules say. `extra` ships paths outside the sync
        allowlist that must still reach the host (typically the staged job script) the same
        way. A path shipped by name that vanishes before the stream reaches it fails the whole
        mirror, so a submission never continues on a partial sync. Fails fast when no include
        paths are declared or a required group is incomplete.

        The vendored path dependencies ship as their own scope, each link replaced by what it
        refers to (see `engines.compile.vendor`): on the machine that has the source that tree
        is links into it, which a host could never follow, so it receives the ordinary tree of
        real files its own compile then leaves alone, and a distribution the manifest stopped
        declaring is pruned without reaching anything beside it.

        `ssh` decides where the transfer actually lands. A declared host is its own alias and the
        user's ssh config answers for it; a machine rented for one job has no alias at all, so a
        policy bound to that machine names it instead and the same mirror reaches a box this
        workspace had never heard of a minute ago.
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
        incomplete = [
            list(group)
            for group in required
            if not all(self.local(path).is_file() for path in group)
        ]
        if incomplete:
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
                vendored = Scope([vendor_root()], deny=patterns(_VENDOR_EXCLUDE), follow=True)
                scopes.append(vendored)
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
        sampler: str = "",
        attestation: str = "",
        containerize: Callable[[list[str]], list[str]] | None = None,
        watch: Watcher | None = None,
        prefix: str = "",
        artifact: Sequence[str] = (),
    ) -> Handle:
        """Render `shipment` into a job script for `plan`'s host and dispatch it.

        Renders a PBS or bash job script (whichever `plan.profile.kind` calls for), wraps the
        command in the container runtime when `plan.containerized`, submits it, and returns a
        `Handle`.

        The script activates and runs from the snapshot of the mirror this dispatch pins, not
        from the mirror itself, so a later dispatch of a different tree cannot change the code
        underneath a job that is already queued or running. The handle still names the mirror,
        which is where the logs and the results are and where they stay once the snapshot is
        pruned.

        plan: the resolved execution context (host, profile, env, container).
        shipment: what the job runs and what it ships, its provenance read once.
        root: the workspace root on `plan.host`.
        resources: the scheduler resource request (queue/walltime/mem/gpus already resolved).
        verify: a preflight command proving the host's activated environment actually runs.
        fetch: a results path recorded on the handle, pulled back by `fetch`.
        node: the ledger slug this run serves, recorded on the run and its receipts.
        gpu_in_select: whether a PBS GPU request belongs in the `select=` chunk.
        sampler: a shell line the script runs beside the command, empty for none. Opaque here
            on purpose, since what a host watches about itself is not the dispatcher's decision.
        attestation: a shell line the script runs in the foreground before the command, empty for
            none, recording what the machine looked like as the work started.
        containerize: builds the container runtime argv around `["bash", "-c", cmd]`; required
            when `plan.containerized`.
        watch: announces the building of the host's environment, the one stage of a dispatch
            that happens on the far side and takes long enough to be worth saying.
        prefix: the built environment this job activates on the host, addressed by the content
            of the compiled artifact this dispatch ships; empty leaves the job activating the
            workspace's own environment, which is what a workspace addressing none still does.
        artifact: the compiled manifest, lock and state this dispatch addressed its environment
            by, shipped with the mirror so the tree it pins carries the very compile the pin was
            taken over. See `_raise_required_sync_failure`'s neighbours for why a group under
            the generated tree has to be named to travel at all.
        """
        container_command = ""
        if plan.containerized:
            if containerize is None:
                raise LookupError(
                    f"plan for host {plan.host!r} is containerized but no container argv "
                    "builder was given"
                )
            container_command = shlex.join(containerize(["bash", "-c", shipment.command]))
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
            container_command=container_command,
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
                # A need can already live on the host without a local copy. The sealed
                # snapshot checks every need on that mirror before dispatch; only local
                # files need punching through the transfer filters here.
                *([need] for need in shipment.needs if self.local(need).is_file()),
                *([pin] for pin in shipment.pins),
            ],
        )
        return Handle(
            id=handle, host=plan.host, root=root, kind=plan.profile.kind, fetch_path=fetch
        )

    def state(self, handle: Handle) -> JobState:
        """One scheduler probe of `handle`, raising `HostUnreachable` when the host is down.

        The unabsorbed form of `probe`, for a caller that has to say which host could not be
        reached and why (a durable sweep reporting a dead host once and moving on) rather than
        quietly retrying it on the next tick.
        """
        with connection(handle.host) as remote:
            scheduler = registry.SCHEDULERS.select(handle.kind, default="ssh")
            return scheduler.state(remote, handle.root, handle=handle.id)

    def states(self, handles: Sequence[Handle]) -> dict[str, JobState]:
        """Every handle's state, keyed by id, over one connection to the host they share.

        The batched twin of `state`, for a caller holding many handles on one host. A dispatch
        cache that has been accumulating for months holds a thousand runs on a single box, and
        asking that box once instead of a thousand times is the difference between a sweep that
        finishes and one nobody waits for. Every handle must name the same host, root and kind,
        since one connection and one scheduler answer for all of them, so the first handle is
        what those are read from.

        A backend whose batched listing does not cover a handle (a SLURM `squeue` that only spans
        live jobs, a PBS server that purged its history) is asked about that handle on its own
        over the same connection, which is where the job's real ending is. `HostUnreachable`
        surfaces exactly as it does from `state`, since a host that will not answer is a fact
        about the host rather than about any one of these jobs.
        """
        if not handles:
            return {}
        shared = handles[0]
        scheduler = registry.SCHEDULERS.select(shared.kind, default="ssh")
        ids = list(dict.fromkeys(handle.id for handle in handles))
        resolved: dict[str, JobState] = {}
        with connection(shared.host) as remote:
            listed = scheduler.states(remote, shared.root, ids)
            for job_id in ids:
                found = listed.get(job_id)
                if found is None:
                    found = scheduler.state(remote, shared.root, handle=job_id)
                resolved[job_id] = found
        return resolved

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
        would reject fails at once instead of after a round trip. `verify` then proves the
        host's activated environment can run a command at all, turning a broken remote env into
        a clear diagnosis before the scheduler ever sees the job. The dispatched run is recorded
        with its git provenance, so a later poll resolves it without re-deriving anything.

        Between the two, the mirror this transfer just brought up to date is snapshotted and the
        scheduler is pointed at that snapshot rather than at the mirror. The mirror is what stays
        cheap to sync and what every log read, results pull and post-mortem keeps addressing;
        the snapshot is what the job runs from, and it is immutable, so the next dispatch of
        another tree rewrites the mirror under nobody.

        shipment: what the job runs and ships, its provenance read once so two dispatches of
            one tree share one snapshot. A caller submitting a hand-written script passes none
            and ships the mirror under the workspace's own provenance.
        listing: the staged closure listing a sealed shipment's snapshot copies from, shipped
            beside the script; empty for a command that ships the mirror.
        containerize: builds the container runtime argv around `["bash", "-c", verify]`; required
            when `plan.containerized`, so the verify preflight runs inside the same base image a
            job would.
        watch: announces the one stage this has that nobody can see from here, the building of
            the environment on the host.
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
        """Render `spec`, write it under `{STATE_DIR}/jobs/`, return its workspace-relative path.

        The file is content-addressed, so repeated runs reuse it instead of growing the jobs
        directory unboundedly. The path comes back relative to the workspace because it is also
        what the mirror recreates on the host and what the scheduler is told to run there.
        """
        text = spec.render(pbs=pbs, gpu_in_select=gpu_in_select)
        digest = hashlib.sha256(text.encode()).hexdigest()
        return self._stage(f"job-{digest}.sh", text.encode())

    @staticmethod
    def _bare_name_or_raise(script: str, error: FileNotFoundError) -> str:
        """`script` unchanged when it is a bare name, else re-raise with staging detail."""
        if Path(script).name == script:
            return script
        raise FileNotFoundError(
            f"cannot submit script {script!r}: the local file does not exist, so it "
            "cannot be shipped to the host"
        ) from error

    def _prepare_script(self, script: str) -> tuple[str, tuple[str, ...]]:
        """Stage a concrete local script and return its host-safe path plus its sync source.

        A bare name stays unchanged (a future on-host executor resolves it in-repository). An
        explicit path must exist locally, since forwarding an unresolved local path would make
        the host fail later with no way to guarantee what it runs.
        """
        try:
            content = self.local(script).read_bytes()
        except FileNotFoundError as error:
            return Dispatcher._bare_name_or_raise(script, error), ()
        digest = hashlib.sha256(content).hexdigest()
        staged = self._stage(f"job-{digest}.sh", content)
        return staged, (staged,)

    def _protected_outputs(self, fetch: str, *, sources: Sequence[str] = ()) -> list[str]:
        """Download-only literal output paths, excluding conflicts with explicitly shipped input.

        The existing workspace cache spans host aliases and retains completed evidence. Each
        path is literal, so glob characters in a real filename never widen what it protects.
        """
        paths = {run.fetch_path for run in self.cache.recent(limit=None) if run.fetch_path}
        paths.update([fetch] if fetch else [])
        outputs = []
        for path in sorted(paths):
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
        """Persist a terminal state to the cache and project it onto a `Verdict`.

        Reads the host's log for the failure reason only when the job did not end `ok`, so a
        clean run never pays a second round-trip.
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
        """Build the pinned environment before submitting; refuse a failed build.

        remote: the open connection to the host.
        plan: the resolved execution context; containers need no separate prefix.
        pinned: the snapshot the wave will run out of.
        root: the host workspace containing the built environments.
        watch: receives a successful build announcement.
        prefix: the expected environment address.
        """
        command = providing(plan, root=root, pinned=pinned, prefix=prefix)
        if not command:
            return
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
        """Fail fast, in one plain sentence, when `plan.host`'s activated environment is broken.

        Runs `verify` through the same activation wrap every job depends on, turning a broken
        env (a stale install, a dependency the sync never shipped) into a clear diagnosis before
        the scheduler ever sees the job, instead of a raw traceback buried inside its log.
        """
        body = wrap(plan, root, command=verify, containerize=containerize)
        retcode, _, err = remote["bash"][["-lc", body]].run(retcode=None)
        if retcode != 0:
            raise SystemExit(f"environment on {plan.host!r} is broken: {failure_reason(err)}")
