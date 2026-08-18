# Render a scheduler job script from a single command, so users stop hand-writing one shell
# script per experiment. `JobSpec` is the value object; the script text lives in jinja
# templates under `templates/`.

import shlex

from jinja2 import Environment, PackageLoader
from patos import FrozenModel

from ...context.plan import ExecutionPlan
from ..shared import state_dir
from ..wrapping import activation_stage

# The job script templates shipped with the package. `trim_blocks`/`lstrip_blocks` make the
# `{% %}` control lines vanish from the rendered shell text; `keep_trailing_newline` keeps the
# scripts newline-terminated like any shell file.
_TEMPLATES = Environment(  # ruff:ignore[jinja2-autoescape-false]  reason=renders shell scripts, not HTML; autoescape would corrupt them since=2026-08-16
    loader=PackageLoader(__package__, "templates"),
    keep_trailing_newline=True,
    trim_blocks=True,
    lstrip_blocks=True,
)


class JobSpec(FrozenModel):
    """One job: a command plus the knobs its rendered script needs.

    Walltime semantics differ by backend, deliberately. A PBS queue always enforces a walltime,
    so `render(pbs=True)` requires one, resolved by the caller from the host's queue defaults
    (never invented here). A schedulerless host (pueue/bash/slurm) enforces a cap only when the
    caller explicitly chose one: an invisible default that kills correct work is worse than a
    hung job a monitor can see and cancel. When a cap is set, the wrapper stamps `mainboard:
    killed at walltime HH:MM:SS` into the log so a triage view decodes the stop.

    cmd: the command to run (e.g. `python -m experiments.x.run --model X`).
    plan: the resolved execution context, which names the host and the environment the script
        activates. Carrying the plan rather than a prefix and an environment name separately is
        what makes it impossible to render a script whose prefix and environment disagree.
    root: the workspace root on the host, so the script activates through the same generated
        activation a wrapped command does.
    queue/select/gpus/account/mem_gb: PBS header values (ignored when rendering a bash wrapper).
    walltime: `HH:MM:SS` cap; empty means the bare `#PBS` requirement is unmet (a PBS render
        raises) or, on a schedulerless host, that the job runs uncapped.
    pythonpath: explicit `PYTHONPATH` the job runs under, empty for an isolated default.
    isolate_pythonpath: drop whatever `PYTHONPATH` the submitting shell exported, so a job's
        imports come only from its own environment; False keeps the inherited value for a
        caller that deliberately relies on it.
    container_command: a preformatted shell command that already wraps the inner `cmd` in a
        container runtime invocation; when set, the body runs this instead of a bare `bash -c`.
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
    container_command: str = ""

    def render(self, *, pbs: bool, gpu_in_select: bool = True) -> str:
        """The job script text: a full PBS script when `pbs`, else a bash wrapper.

        The PBS header always carries `-j oe` and redirects merged output directly into
        `{STATE_DIR}/logs/<bare jobid>.log`. Its exit trap appends the final status after that
        output and writes the same status to `{STATE_DIR}/logs/<bare jobid>.exit` so a job the
        server later purges can still be autopsied.

        `ngpus` joins the `select=` chunk only when `gpus` > 0 and `gpu_in_select`; some GPU
        queues hand the GPU out with the queue and reject an explicit `ngpus`, so such a host
        passes `gpu_in_select=False`. `mem=NNgb` joins the same chunk when `mem_gb` is set.

        pbs: render the PBS header script instead of the bash wrapper.
        gpu_in_select: whether a GPU request belongs in the `select=` chunk.
        """
        if pbs and not self.walltime:
            raise ValueError(
                "a PBS job needs an explicit walltime; resolve one from the host's queue "
                "defaults before rendering"
            )
        template = _TEMPLATES.get_template("pbs_job.sh.j2" if pbs else "bash_job.sh.j2")
        ngpus = f":ngpus={self.gpus}" if self.gpus and gpu_in_select else ""
        mem = f":mem={self.mem_gb}gb" if self.mem_gb else ""
        chunk = f"select={self.select}{ngpus}{mem}"
        return template.render(
            cmd=shlex.quote(self.cmd),
            queue=self.queue,
            walltime=self.walltime,
            walltime_seconds=walltime_seconds(self.walltime) if self.walltime else 0,
            chunk=chunk,
            account=self.account,
            pythonpath=shlex.quote(self.pythonpath) if self.pythonpath else "",
            isolate_pythonpath=self.isolate_pythonpath,
            activation=activation_stage(self.plan, self.root),
            container_command=self.container_command,
            state_dir=state_dir(),
        )


def walltime_seconds(walltime: str) -> int:
    """A `HH:MM:SS` walltime as whole seconds, for the bash host's `timeout` wrapper."""
    hours, minutes, seconds = (int(part) for part in walltime.split(":"))
    return hours * 3600 + minutes * 60 + seconds
