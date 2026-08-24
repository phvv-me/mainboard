# The SLURM backend: submit through `sbatch`, monitor with `squeue`/`sacct`. Every command
# is built by pure functions here, keeping the backend unit-testable without a live cluster.

import re
import shlex
from enum import StrEnum
from typing import TYPE_CHECKING

from patos import Model

from ..shared import state_dir
from ..vocabulary import JobState, Resources
from .base import read_log

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..transport import Machine


class SlurmState(StrEnum):
    """SLURM job states reported by `squeue`/`sacct` (the long-form names).

    `sacct` also appends a node reason in parentheses for `CANCELLED by <uid>`, stripped before
    lookup.
    """

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUSPENDED = "SUSPENDED"
    COMPLETING = "COMPLETING"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"
    TIMEOUT = "TIMEOUT"
    NODE_FAIL = "NODE_FAIL"
    OUT_OF_MEMORY = "OUT_OF_MEMORY"
    BOOT_FAIL = "BOOT_FAIL"
    DEADLINE = "DEADLINE"
    PREEMPTED = "PREEMPTED"


# States in which a job is still in flight (not a terminal verdict).
SLURM_LIVE = {SlurmState.PENDING, SlurmState.RUNNING, SlurmState.SUSPENDED, SlurmState.COMPLETING}

_SQUEUE_FORMAT = "%i|%j|%T|%P|%M"
_SACCT_FORMAT = "JobID,State,ExitCode"

# `%j` is SLURM's own job-id substitution in `--output`; same merged-output convention the
# PBS/bash job templates write to.
_LOG_TEMPLATE = f"{state_dir()}/logs/%j.log"


class SlurmJob(Model):
    """One SLURM job row, parsed from `squeue` or `sacct`."""

    job_id: str
    name: str = ""
    state: SlurmState | str
    exit_code: int | None = None
    partition: str | None = None
    elapsed: str | None = None


def parse_slurm_state(value: str) -> SlurmState | str:
    """Parse a SLURM state token, dropping any `CANCELLED by 1000` suffix."""
    head = value.strip().split(" ", maxsplit=1)[0].upper()
    try:
        return SlurmState(head)
    except ValueError:
        return value.strip()


def parse_exit_code(value: str) -> int | None:
    """Parse `sacct`'s `ExitCode` field (`<returncode>:<signal>`).

    Returns the return code, or the signal number when the job was killed by a signal (return
    code 0 but signal non-zero). None when unparsable/empty.
    """
    field = value.strip()
    if not field:
        return None
    code, _, signal = field.partition(":")
    if code.isdigit() and int(code) != 0:
        return int(code)
    if signal.isdigit() and int(signal) != 0:
        return int(signal)
    return int(code) if code.isdigit() else None


def build_squeue_command(*, me: bool = True) -> list[str]:
    """Build a `squeue` command emitting the pipe-delimited format."""
    command = ["squeue", "--noheader", f"--format={_SQUEUE_FORMAT}"]
    if me:
        command.append("--me")
    return command


def parse_squeue_output(output: str) -> list[SlurmJob]:
    """Parse the pipe-delimited `squeue` output into `SlurmJob` rows."""
    jobs: list[SlurmJob] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        parts = line.split("|")
        if len(parts) < 5:
            continue
        job_id, name, state, partition, elapsed = (part.strip() for part in parts[:5])
        jobs.append(
            SlurmJob(
                job_id=job_id,
                name=name,
                state=parse_slurm_state(state),
                partition=partition or None,
                elapsed=elapsed or None,
            )
        )
    return jobs


def build_sacct_command(job_id: str) -> list[str]:
    """Build the `sacct` post-mortem command for one job."""
    return ["sacct", "--jobs", job_id, f"--format={_SACCT_FORMAT}", "--parsable2", "--noheader"]


def parse_sacct_output(output: str, *, job_id: str) -> SlurmJob | None:
    """Parse `sacct` output for `job_id` into a `SlurmJob`, keeping the top-level `<id>` row.

    Returns None when the job is absent from the accounting database (vanished). Sub-steps
    (`<id>.batch`, `<id>.extern`) are ignored.
    """
    for line in output.splitlines():
        if not line.strip():
            continue
        parts = line.split("|")
        if len(parts) < 3:
            continue
        row_id, state, exit_code = (part.strip() for part in parts[:3])
        if row_id != job_id:
            continue
        return SlurmJob(
            job_id=row_id, state=parse_slurm_state(state), exit_code=parse_exit_code(exit_code)
        )
    return None


def build_resource_flags(resources: Resources) -> list[str]:
    """Render `resources` as the allocation flags `sbatch` and `srun` both take.

    `gpus` is only emitted when set, so CPU-only jobs run on clusters without GPU GRES.
    """
    flags: list[str] = []
    if resources.gpus:
        flags.append(f"--gpus={resources.gpus}")
    if resources.walltime is not None:
        flags.append(f"--time={resources.walltime}")
    if resources.queue is not None:
        flags.append(f"--partition={resources.queue}")
    if resources.account:
        flags.append(f"--account={resources.account}")
    if resources.mem_gb is not None:
        flags.append(f"--mem={resources.mem_gb}G")
    return flags


def build_sbatch_flags(resources: Resources, script: str) -> list[str]:
    """Render `resources` as `sbatch` flags, including the output sink and the script itself."""
    return ["sbatch", f"--output={_LOG_TEMPLATE}", *build_resource_flags(resources), script]


def _extract_job_id(output: str) -> str:
    """Pull the job id out of `sbatch` output (`Submitted batch job 12345`)."""
    if match := re.search(r"Submitted batch job\s+(\d+)", output):
        return match.group(1)
    return output.strip().splitlines()[-1].strip() if output.strip() else ""


def slurm_verdict(state: SlurmState | str | None, exit_code: int | None) -> str:
    """A one-word verdict for a SLURM job from its `sacct` state and exit code."""
    if state is None:
        return "vanished"
    if state in SLURM_LIVE:
        return "running"
    if state == SlurmState.COMPLETED and (exit_code or 0) == 0:
        return "ok"
    return "failed"


class Slurm:
    """Dispatch jobs to a SLURM cluster via `sbatch`."""

    name = "slurm"

    def cancel(self, remote: Machine, root: str, *, handle: str) -> None:
        del root
        remote["bash"][["-lc", f"scancel {shlex.quote(handle)}"]](retcode=None)

    def interactive(self, *, env: str, command: Sequence[str], resources: Resources) -> str:
        """An interactive SLURM allocation, `srun --pty` under the batch allocation flags.

        Unlike PBS, `srun` takes the command to run on the allocated node, so a probe rides the
        same allocation an empty command hands a login shell. The environment is activated from
        inside the session, since nothing this side of `srun` runs on the allocated node.
        """
        del env
        shell = ["bash", "-l"]
        return shlex.join(["srun", "--pty", *build_resource_flags(resources), *(command or shell)])

    def logs(self, remote: Machine, root: str, *, handle: str) -> str:
        return read_log(remote, root, handle=handle)

    def state(self, remote: Machine, root: str, *, handle: str) -> JobState:
        del root
        output = self.__cluster_command(remote, build_sacct_command(handle))
        job = parse_sacct_output(output, job_id=handle)
        state = job.state if job else None
        exit_code = job.exit_code if job else None
        return JobState(
            handle=handle,
            state=str(state) if state is not None else None,
            exit_code=exit_code,
            verdict=slurm_verdict(state, exit_code),
        )

    def states(self, remote: Machine, root: str, handles: Sequence[str]) -> dict[str, JobState]:
        """Every job `squeue` still lists for this user, keyed by handle.

        One listing covers the whole cluster's live jobs, so a caller holding many handles on
        this host pays a single round trip. A handle `squeue` no longer carries is left absent,
        since only `sacct` knows how a finished job ended, and the caller falls back to the
        per-handle `state` for it.
        """
        del root, handles
        output = self.__cluster_command(remote, build_squeue_command(me=True))
        return {
            job.job_id: JobState(
                handle=job.job_id,
                label=job.name,
                state=str(job.state),
                verdict=slurm_verdict(job.state, None),
            )
            for job in parse_squeue_output(output)
        }

    def submit(
        self,
        remote: Machine,
        root: str,
        *,
        script: str,
        args: Sequence[str],
        resources: Resources,
    ) -> str:
        del args  # SLURM scripts are self-contained; sbatch takes no free-form positional args.
        command = shlex.join(build_sbatch_flags(resources, script))
        retcode, out, err = remote["bash"][["-lc", command]].run(retcode=None)
        handle = _extract_job_id(out)
        if not handle.isdigit():
            raise SystemExit(
                f"sbatch failed (rc={retcode}): {(err or out).strip()[-400:] or '(no output)'}"
            )
        return handle

    def __cluster_command(self, remote: Machine, command: list[str]) -> str:
        """Run a built `squeue`/`sacct`/`sinfo` argv under a login shell, returning its stdout."""
        return str(remote["bash"][["-lc", shlex.join(command)]](retcode=None))
