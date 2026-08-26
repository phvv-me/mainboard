# The CLI-free core of a dispatch: submit a job and get a handle back. `Dispatcher` holds the
# reusable core every dispatch shares, and hands back a `Handle` a caller can poll, await, or
# fetch.

import hashlib
import shlex
import subprocess  # ruff:ignore[suspicious-subprocess-import]  reason=fixed local invocation off PATH, not untrusted input since=2026-08-18
from collections.abc import (
    Sequence,  # ruff:ignore[typing-only-standard-library-import]  reason=await_many is inspect.signature()'d in tests, so its Sequence[Handle] annotation must resolve at runtime since=2026-08-17
)
from contextlib import suppress
from math import ceil
from pathlib import Path
from time import sleep
from typing import TYPE_CHECKING

from patos import FrozenModel
from plumbum.commands.processes import ProcessExecutionError

from ..context.admission import admit
from . import vocabulary
from .jobs import JobSpec
from .schedulers import HostUnreachable, failure_reason, pick, read_log, registry
from .shared import HandleId, db_file, logger, now, state_path, workspace
from .state.cache import Cache, RunRecord
from .sync import GitignoreFilter, SyncLock, rsync
from .sync import Rsync as RsyncFlags
from .transport import SshTransport
from .vocabulary import JobState, Resources
from .wrapping import connection, wrap

if TYPE_CHECKING:
    from collections.abc import Callable

    from ..context.plan import ExecutionPlan
    from .transport import Machine

# How a finished verdict maps to a process exit code: 0 ok, 1 failed, 2 still running, 3
# vanished/unknown. A caller can branch on this without re-deriving it.
_VERDICT_EXITS = {"ok": 0, "failed": 1, "running": 2}


def git(*args: str) -> str:
    """Stripped stdout of a local `git` command (the dispatched run's provenance)."""
    argv = ["git", *args]  # fixed local invocation off PATH, not untrusted input
    return subprocess.run(argv, capture_output=True, text=True, check=False).stdout.strip()  # ruff:ignore[subprocess-without-shell-equals-true]  reason=fixed local invocation off PATH, not untrusted input since=2026-08-16


class Handle(FrozenModel):
    """A dispatched job, enough to poll, await, or fetch it without re-resolving the host.

    id: the scheduler's job handle (PBS job id, pueue task id, SLURM job id, the local run id),
        always text: pueue hands out small integers and a caller who read one back as a number
        would otherwise fail validation deep inside a status poll.
    host: the ssh alias the job runs on, or the declared alias of the provider host it was
        rented for.
    root: the workspace root on that host, empty for a provider that syncs no workspace.
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
        """rsync the handle's recorded results path back from its host into the same local path."""
        if not handle.fetch_path:
            raise LookupError(f"handle {handle.id!r} has no fetch path to pull")
        self.fetch_path(handle.host, root=handle.root, path=handle.fetch_path, ssh=ssh)

    def fetch_path(
        self, host: str, *, root: str, path: str, ssh: SshTransport | None = None
    ) -> None:
        """rsync `path` back from `host` into the same workspace path (a file or a directory)."""
        policy = ssh or SshTransport()
        target = self.local(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        rsync(
            [f"{host}:{root}/{Path(path)}"],
            f"{target.parent}/",
            rsh=policy.rsync_shell,
            timeout=ceil(policy.deadline),
            host=host,
        )
        logger.info("fetched %s from %s", path, host)

    def local(self, path: str) -> Path:
        """`path` as a real file here: a workspace-relative name resolved against the root.

        The one place a dispatch turns a written-down path into a file on this machine, so a
        command typed in a subdirectory reads and writes the same files it would from the
        workspace root. An absolute path is already a location and passes through.
        """
        given = Path(path).expanduser()
        return given if given.is_absolute() else self.root / given

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

    def rsync_up(
        self,
        plan: ExecutionPlan,
        root: str,
        *,
        ssh: SshTransport | None = None,
        required: Sequence[Sequence[str]] = (),
        extra: Sequence[str] = (),
    ) -> None:
        """Mirror the workspace to `plan.host`; git-ignored files and the denylist skipped.

        The workspace and nested `.gitignore` files are the primary send and delete boundary;
        `plan.profile.sync.protect` is the escape hatch for remote-only artifacts outside that
        boundary. `required` names groups of paths that must ship together despite being
        outside the allowlist or git-ignored (a compiled manifest with its lock and the state
        naming what that lock was solved from, say): each group is required to exist locally as
        a whole, and is punched through the denylist with its own include filter. `extra` ships
        paths outside the sync allowlist that must still reach the host (typically the staged
        job script). Fails fast when no include paths are declared or a required group is
        incomplete.
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
        directories = dict.fromkeys(Path(path).parts[0] for group in required for path in group)
        include_filters = [
            *(f"/{directory}/" for directory in directories),
            *(f"/{path}" for group in required for path in group),
        ]
        remainder_filters = [f"/{directory}/***" for directory in directories]
        gitignore_files = self.sync.control_files(include)
        required_paths = list(dict.fromkeys(path for group in required for path in group))
        with SyncLock(plan.host, self.sync.root):
            try:
                rsync(
                    [*include, *gitignore_files, *required_paths, *extra],
                    f"{plan.host}:{root}/",
                    RsyncFlags.ARCHIVE
                    | RsyncFlags.COMPRESS
                    | RsyncFlags.RELATIVE
                    | RsyncFlags.VERBOSE
                    | RsyncFlags.DELETE
                    | RsyncFlags.DELETE_AFTER,
                    include=include_filters,
                    filters=self.sync.filters,
                    exclude=[*remainder_filters, *self.sync.excludes, *scope.exclude],
                    protect=scope.protect,
                    rsh=policy.rsync_shell,
                    timeout=ceil(policy.deadline),
                    host=plan.host,
                    allow_vanished=not gitignore_files and not required_paths and not extra,
                    cwd=self.root,
                )
            except ProcessExecutionError as error:
                Dispatcher._raise_required_sync_failure(
                    error, plan.host, required_paths, extra=extra
                )
        self.cache.mark_synced(plan.host)

    def run(
        self,
        plan: ExecutionPlan,
        cmd: str,
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
    ) -> Handle:
        """Render `cmd` into a job script for `plan`'s host and dispatch it.

        Renders a PBS or bash job script (whichever `plan.profile.kind` calls for), wraps `cmd`
        in the container runtime when `plan.containerized`, submits it, and returns a `Handle`.

        plan: the resolved execution context (host, profile, env, container).
        cmd: the command the generated job runs.
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
        """
        container_command = ""
        if plan.containerized:
            if containerize is None:
                raise LookupError(
                    f"plan for host {plan.host!r} is containerized but no container argv "
                    "builder was given"
                )
            container_command = shlex.join(containerize(["bash", "-c", cmd]))
        spec = JobSpec(
            cmd=cmd,
            plan=plan,
            root=root,
            queue=resources.queue or "",
            walltime=resources.walltime or "",
            select=resources.nodes,
            gpus=resources.gpus,
            account=resources.account,
            mem_gb=resources.mem_gb,
            container_command=container_command,
            sampler=sampler,
            attestation=attestation,
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
            containerize=containerize,
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
        containerize: Callable[[list[str]], list[str]] | None = None,
    ) -> str:
        """Ship the workspace, admit the request, dispatch `script`, and return the handle.

        Admission runs before any ssh connection, so a request the queue's declared policy
        would reject fails at once instead of after a round trip. `verify` then proves the
        host's activated environment can run a command at all, turning a broken remote env into
        a clear diagnosis before the scheduler ever sees the job. The dispatched run is recorded
        with its git provenance, so a later poll resolves it without re-deriving anything.

        containerize: builds the container runtime argv around `["bash", "-c", verify]`; required
            when `plan.containerized`, so the verify preflight runs inside the same base image a
            job would.
        """
        admit(
            plan.profile,
            queue=resources.queue or "",
            walltime=resources.walltime or "",
            mem_gb=resources.mem_gb or 0,
        )
        prepared, staged = self._prepare_script(script)
        self.rsync_up(plan, root, required=required, extra=staged)
        sha = git("rev-parse", "--short", "HEAD")
        dirty = bool(git("status", "--porcelain"))
        with connection(plan.host) as remote:
            self._verify(remote, plan, root, verify=verify, containerize=containerize)
            try:
                handle = pick(plan.profile).submit(
                    remote, root, script=prepared, args=args, resources=resources
                )
            except SystemExit as error:
                raise SystemExit(f"submission to host {plan.host!r} failed: {error}") from None
        self.cache.record(
            RunRecord(
                handle=handle,
                target=plan.host,
                kind=plan.profile.kind,
                script=prepared,
                args=" ".join(shlex.quote(a) for a in args),
                git_sha=sha,
                dirty=int(dirty),
                submitted_at=now(),
                fetch_path=fetch,
                name=name,
                node=node,
            )
        )
        logger.info(
            "%s -> %s on %s (%s%s)", prepared, handle, plan.host, sha, "+dirty" if dirty else ""
        )
        return handle

    def track(
        self,
        handle: str,
        *,
        host: str,
        kind: str,
        command: str,
        name: str = "",
        node: str = "",
        fetch: str | None = None,
    ) -> Handle:
        """Record a provider-dispatched run in the shared cache and return its `Handle`.

        A provider backend owns its own transport, so none of the shipping, verifying and
        scheduling `submit` does applies to it, but the run still has to land in the same cache
        every other dispatch does or no later process can settle it. That matters more here than
        for a queue, since an untracked rental keeps billing after its command ends and being
        tracked is what lets the durable sweep end it.

        handle: the provider's own opaque run id.
        host: the alias the run was dispatched to.
        kind: the provider kind, which is how a later pass finds the backend again.
        command: the command the run was launched with, kept as its provenance.
        name: a human label for the run, a study's label when a study owns it.
        node: the ledger slug this run serves, recorded on the run and its receipts.
        fetch: a results path recorded for a later pull.
        """
        self.cache.record(
            RunRecord(
                handle=handle,
                target=host,
                kind=kind,
                script=command,
                args="",
                git_sha=git("rev-parse", "--short", "HEAD"),
                dirty=int(bool(git("status", "--porcelain"))),
                submitted_at=now(),
                fetch_path=fetch,
                name=name,
                node=node,
            )
        )
        logger.info("%s -> %s on %s (%s)", command, handle, host, kind)
        return Handle(id=handle, host=host, root="", kind=kind, fetch_path=fetch)

    def write_job_script(self, spec: JobSpec, *, pbs: bool, gpu_in_select: bool = True) -> str:
        """Render `spec`, write it under `{STATE_DIR}/jobs/`, return its workspace-relative path.

        The file is content-addressed, so repeated runs reuse it instead of growing the jobs
        directory unboundedly. The path comes back relative to the workspace because it is also
        what the mirror recreates on the host and what the scheduler is told to run there.
        """
        text = spec.render(pbs=pbs, gpu_in_select=gpu_in_select)
        digest = hashlib.sha256(text.encode()).hexdigest()[:12]
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

    @staticmethod
    def _raise_required_sync_failure(
        error: ProcessExecutionError,
        host: str,
        required_paths: Sequence[str],
        *,
        extra: Sequence[str],
    ) -> None:
        """Re-raise `error` verbatim when nothing required was in flight, else wrap it."""
        if not required_paths and not extra:
            raise error
        paths = ", ".join((*required_paths, *extra))
        raise RuntimeError(
            f"failed to ship required sync path(s) {paths} to {host}; "
            "submission aborted before scheduler dispatch"
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
        digest = hashlib.sha256(content).hexdigest()[:12]
        staged = self._stage(f"job-{digest}.sh", content)
        return staged, (staged,)

    def _stage(self, name: str, content: bytes) -> str:
        """Write `content` into the jobs directory and answer its workspace-relative path."""
        path = state_path(self.root) / "jobs" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return str(path.relative_to(self.root))

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
