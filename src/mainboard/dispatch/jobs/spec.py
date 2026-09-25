# Render a scheduler job script from a single command, so users stop hand-writing one shell
# script per experiment. `JobSpec` is the value object. What the script does is a `runtime.Job`
# record the host's own tool carries out; the script itself is only the handover, since a
# scheduler still wants a file it can run and PBS reads its directives from that file's header.

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

    Walltime semantics differ by backend, deliberately. A PBS queue always enforces a walltime,
    so `render(pbs=True)` requires one, resolved by the caller from the host's queue defaults
    (never invented here). A schedulerless host (pueue/bash/slurm) enforces a cap only when the
    caller explicitly chose one: an invisible default that kills correct work is worse than a
    hung job a monitor can see and cancel. When a cap is set, the runner stamps `mainboard:
    killed at walltime HH:MM:SS` into the log so a triage view decodes the stop.

    cmd: the command to run (e.g. `python -m experiments.x.run --model X`).
    plan: the resolved execution context, which names the host and the environment the job
        enters. Carrying the plan rather than a prefix and an environment name separately is
        what makes it impossible to render a job whose prefix and environment disagree.
    root: the tree on the host the job runs from, which is the snapshot of the mirror this
        dispatch pinned. Its `.mainboard/` is a symlink back to the mirror, so activating
        through it hands the job the mirror's environment while its code stays frozen.
    queue/select/gpus/account/mem_gb: PBS header values (ignored when rendering a plain script).
    walltime: `HH:MM:SS` cap; empty means the bare `#PBS` requirement is unmet (a PBS render
        raises) or, on a schedulerless host, that the job runs uncapped.
    pythonpath: explicit `PYTHONPATH` the job runs under, empty for an isolated default. A
        dispatch fills it with the pinned tree's own import roots, so a package this workspace
        installs editable is imported from the snapshot the job was frozen at rather than from
        the mirror the shared prefix's editable install points at.
    isolate_pythonpath: drop whatever `PYTHONPATH` the submitting shell exported, so a job's
        imports come only from its own environment; False keeps the inherited value for a
        caller that deliberately relies on it. An explicit `pythonpath` replaces the inherited
        value outright and is therefore isolated the same way.
    container: the argv that runs `cmd` inside a container runtime, empty for a bare command.
    prefix: the built environment this job activates, addressed by the content of the manifest
        and lock it was dispatched with. Set, the job enters exactly that directory and asks
        nothing to reconcile it; empty, it enters the workspace's own environment, which is what
        an interactive run and a hand-written script still want.
    provide: builds that environment before the job enters it, when the host does not have it
        yet. None for a job whose environment is an image.
    sampler: watches the host beside the command, None for a host that watches nothing.
    attestation: records what the machine looked like immediately before the command, None for
        none. Ordered before the sampler because a reading taken after the command is under way
        describes the command rather than the conditions.
    source: the captured SHA-256 content identity, exported as `MAINBOARD_SOURCE`.
    commit: historical metadata only; new dispatches leave this empty.
    digest: that tree's content digest, exported as `MAINBOARD_SOURCE_DIGEST`. A preflight on a
        mirror verifies this digest against the listing and the actual source bytes.
    closure: where the job finds its closure listing, exported as `MAINBOARD_CLOSURE`, so a
        receipt can list what it ran on and the runner can refuse an import outside it. Empty
        for a command that ships the mirror.
    first_party: the top-level names the workspace's own import roots define, colon-joined,
        exported as `MAINBOARD_FIRST_PARTY` beside the closure for the runner's finder.
    deferred: top-level names whose whole distribution the closure left to the environment,
        colon-joined, exported as `MAINBOARD_DEFERRED` so the runner's finder admits them.
    exports: the host profile's `[hosts.<name>.exports]`, set after everything else so every
        job on that host runs in the world its profile declares.
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
        """The job script text: the `#PBS` header when `pbs`, then the handover to the tool.

        The script is POSIX `sh` and does one thing, `exec` the host's own tool on the job
        record, which travels inline so a scheduler that feeds the script to its shell on stdin
        runs it as surely as one that passes its path. The per-user install directories lead
        `PATH` first, since a batch shell is not a login shell and may not have them.

        Under PBS the runner appends the merged output to `{STATE_DIR}/logs/<bare jobid>.log`
        and the final status to `{STATE_DIR}/logs/<bare jobid>.exit`, so a job the server later
        purges can still be autopsied; the header's `-j oe` merges the two streams PBS spools.

        `ngpus` joins the `select=` chunk only when `gpus` > 0 and `gpu_in_select`; some GPU
        queues hand the GPU out with the queue and reject an explicit `ngpus`, so such a host
        passes `gpu_in_select=False`. `mem=NNgb` joins the same chunk when `mem_gb` is set.

        pbs: render the PBS header and hand PBS the output and walltime.
        gpu_in_select: whether a GPU request belongs in the `select=` chunk.
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
        """The record the host's tool runs, the walltime and the output left to PBS under it.

        pbs: the job runs under PBS, which enforces the walltime and spools the output itself.
        """
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
