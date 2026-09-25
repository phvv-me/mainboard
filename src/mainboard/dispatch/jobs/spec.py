# Render a scheduler job script from a single command. What it does is a `runtime.Job` record the
# host's own tool carries out; the script is only the handover, since a scheduler still wants a
# file to run and PBS reads its directives from that file's header.

import shlex

from patos import FrozenModel

from ...context.plan import ExecutionPlan
from ...core.project import Project
from ...runtime.job import Job, PrefixActivation, ToolCall, WorkspaceActivation
from ..shared import (
    CLOSURE_VAR,
    COMMIT_VAR,
    DEFERRED_VAR,
    DIGEST_VAR,
    FIRST_PARTY_VAR,
    SOURCE_VAR,
    state_dir,
)
from ..wrapping import USER_BINS, absent, activation, missing


class JobSpec(FrozenModel):
    """One job: a command plus the knobs its rendered script needs.

    Walltime differs by backend, deliberately. A PBS queue always enforces one, so a PBS render
    requires it, resolved by the caller from the host's queue defaults (never invented here). A
    schedulerless host (pueue/bash/slurm) is capped only when the caller chose a cap, since an
    invisible default that kills correct work is worse than a hung job a monitor can cancel; a
    set cap makes the runner log `mainboard: killed at walltime HH:MM:SS` for triage.

    plan: names the host and the environment the job enters, carried whole so a job's prefix
        and environment can never disagree.
    root: the snapshot of the mirror this dispatch pinned, the job's tree on the host. Its
        `.mainboard/` links back to the mirror, handing the job the mirror's environment while
        its code stays frozen.
    queue/select/gpus/account/mem_gb: PBS header values, ignored by a plain script.
    walltime: `HH:MM:SS` cap, empty for uncapped (a PBS render raises).
    pythonpath: explicit `PYTHONPATH`, empty for an isolated default. A dispatch fills it with
        the pinned tree's import roots, so an editable workspace package is imported from the
        snapshot rather than the mirror the shared prefix's editable install points at.
    isolate_pythonpath: drop the submitting shell's `PYTHONPATH` so imports come only from the
        job's environment; False keeps it for a caller relying on it. An explicit `pythonpath`
        replaces it outright, isolated the same way.
    container: the argv that runs `cmd` inside a container runtime, empty for a bare command.
    prefix: the built environment to activate, addressed by the manifest and lock content it
        was dispatched with, entered as is with nothing asked to reconcile it; empty enters the
        workspace's own environment, as an interactive run and a hand-written script want.
    provide: builds that environment when the host lacks it, None for an image environment.
    sampler: watches the host beside the command, None for none.
    attestation: records the machine immediately before the command, None for none; it runs
        before the sampler, since a reading taken once the command runs describes the command.
    source: the captured SHA-256 content identity, exported as `MAINBOARD_SOURCE`.
    commit: historical metadata only; new dispatches leave this empty.
    digest: the tree's content digest, exported as `MAINBOARD_SOURCE_DIGEST`, which a preflight
        on a mirror verifies against the listing and the actual source bytes.
    closure: the closure listing, exported as `MAINBOARD_CLOSURE` so a receipt lists what it ran
        on and the runner refuses an import outside it; empty when the command ships the mirror.
    first_party: the workspace import roots' top-level names, colon-joined, exported as
        `MAINBOARD_FIRST_PARTY` for the runner's finder.
    deferred: colon-joined top-level names whose whole distribution the closure left to the
        environment, exported as `MAINBOARD_DEFERRED` so the runner's finder admits them.
    exports: the host profile's `[hosts.<name>.exports]`, set last so every job on that host
        runs in the world its profile declares.
    """

    cmd: str
    plan: ExecutionPlan
    root: str
    queue: str = ""
    walltime: str = ""
    select: int = 1
    gpus: int = 0
    account: str = ""
    mem_gb: int | None = None
    pythonpath: str = ""
    isolate_pythonpath: bool = True
    container: tuple[str, ...] = ()
    prefix: str = ""
    provide: ToolCall | None = None
    sampler: ToolCall | None = None
    attestation: ToolCall | None = None
    source: str = ""
    commit: str = ""
    digest: str = ""
    closure: str = ""
    first_party: str = ""
    deferred: str = ""
    exports: dict[str, str] = {}

    def render(self, *, pbs: bool, gpu_in_select: bool = True) -> str:
        """The POSIX `sh` script: the `#PBS` header when `pbs`, then `exec` of the host's tool.

        The job record travels inline, so a scheduler feeding the script on stdin runs it as
        surely as one passing its path, and the per-user install directories lead `PATH`, since
        a batch shell is not a login shell. Under PBS the runner appends the merged output to
        `{STATE_DIR}/logs/<bare jobid>.log` and the status to `.exit` beside it, so a job the
        server later purges can still be autopsied; `-j oe` merges the streams PBS spools.

        pbs: render the PBS header and hand PBS the output and walltime.
        gpu_in_select: put `ngpus` in the `select=` chunk (with `mem=NNgb` when `mem_gb` is set);
            some GPU queues hand out the GPU with the queue and reject an explicit `ngpus`.
        """
        if pbs and not self.walltime:
            raise ValueError(
                "a PBS job needs an explicit walltime; resolve one from the host's queue "
                "defaults before rendering"
            )
        record = shlex.quote(self.job(pbs=pbs).model_dump_json())
        handover = f'PATH="{":".join(USER_BINS)}:$PATH" exec {Project().name} job {record}'
        lines = [
            "#!/bin/sh",
            *(self.directives(gpu_in_select=gpu_in_select) if pbs else ()),
            f"# A {Project().name} job. The record below is what runs; this line hands it over.",
            handover,
        ]
        return "\n".join(lines) + "\n"

    def job(self, *, pbs: bool) -> Job:
        """The record the host's tool runs; under `pbs`, PBS owns the walltime and the output."""
        return Job(
            command=self.cmd,
            root=self.root,
            activation=self.activation(),
            container=self.container,
            walltime="" if pbs else self.walltime,
            logs=f"{self.root}/{state_dir()}/logs" if pbs else "",
            pythonpath=self.pythonpath,
            isolate_pythonpath=self.isolate_pythonpath,
            variables=self.variables(),
            provide=self.provide,
            attestation=self.attestation,
            sampler=self.sampler,
        )

    def activation(self) -> PrefixActivation | WorkspaceActivation:
        """The environment the job enters: its addressed prefix, or the workspace's own."""
        if self.prefix:
            return PrefixActivation(
                prefix=self.prefix, env=self.plan.env, refusal=absent(self.prefix, self.plan.env)
            )
        installed = self.plan.prefix(self.root)
        return WorkspaceActivation(
            script=activation(self.root, env=self.plan.env),
            prefix=installed,
            refusal=missing(self.plan, installed),
        )

    def variables(self) -> dict[str, str]:
        """The provenance the job exports, then the host profile's exports over it."""
        provenance = {
            SOURCE_VAR: self.source,
            COMMIT_VAR: self.commit,
            DIGEST_VAR: self.digest,
            CLOSURE_VAR: self.closure,
            FIRST_PARTY_VAR: self.first_party,
            DEFERRED_VAR: self.deferred,
        }
        return {name: value for name, value in provenance.items() if value} | self.exports

    def directives(self, *, gpu_in_select: bool) -> list[str]:
        """The `#PBS` header lines: queue, chunk, walltime, group and merged output."""
        ngpus = f":ngpus={self.gpus}" if self.gpus and gpu_in_select else ""
        mem = f":mem={self.mem_gb}gb" if self.mem_gb else ""
        return [
            f"#PBS -q {self.queue}",
            f"#PBS -l select={self.select}{ngpus}{mem}",
            f"#PBS -l walltime={self.walltime}",
            *([f"#PBS -W group_list={self.account}"] if self.account else []),
            "#PBS -j oe",
        ]
