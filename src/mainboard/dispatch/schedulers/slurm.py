# The SLURM backend: submit through `sbatch`, monitor with `squeue`/`sacct`, every command built
# by pure functions so the backend is unit-testable without a live cluster.

import re
import shlex
from enum import StrEnum
from typing import TYPE_CHECKING

from patos import Model

from ..shared import state_dir
from ..vocabulary import JobState, Resources
from .base import read_log, within

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..transport import Machine


class SlurmState(StrEnum):
    """SLURM job states reported by `squeue`/`sacct` (the long-form names)."""

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
_SLURM_LIVE = {SlurmState.PENDING, SlurmState.RUNNING, SlurmState.SUSPENDED, SlurmState.COMPLETING}

_SQUEUE_FORMAT = "%i|%j|%T|%P|%M"
_SACCT_FORMAT = "JobID,State,ExitCode"

# `%j` is SLURM's job-id substitution, giving the same merged-output path a PBS runner writes.
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
    """Parse a SLURM state token, dropping the `by <uid>` suffix `sacct` gives `CANCELLED`."""
    head = value.strip().split(" ", maxsplit=1)[0].upper()
    try:
        return SlurmState(head)
    except ValueError:
        return value.strip()


def parse_exit_code(value: str) -> int | None:
    """`sacct`'s `<returncode>:<signal>` as the return code, else a killing signal, else None."""
    code, _, signal = value.strip().partition(":")
    if code.isdigit() and int(code) != 0:
        return int(code)
    if signal.isdigit() and int(signal) != 0:
        return int(signal)
    return int(code) if code.isdigit() else None


def build_squeue_command() -> list[str]:
    """This user's `squeue` listing in the pipe-delimited format."""
    return ["squeue", "--noheader", f"--format={_SQUEUE_FORMAT}", "--me"]


def parse_squeue_output(output: str) -> list[SlurmJob]:
    """Parse the pipe-delimited `squeue` output into `SlurmJob` rows, skipping short lines."""
    rows = ([part.strip() for part in line.split("|")] for line in output.splitlines())
    return [
        SlurmJob(
            job_id=job_id,
            name=name,
            state=parse_slurm_state(state),
            partition=partition or None,
            elapsed=elapsed or None,
        )
        for job_id, name, state, partition, elapsed, *_ in (row for row in rows if len(row) >= 5)
    ]


def build_sacct_command(job_id: str) -> list[str]:
    """Build the `sacct` post-mortem command for one job."""
    return ["sacct", "--jobs", job_id, f"--format={_SACCT_FORMAT}", "--parsable2", "--noheader"]


def parse_sacct_output(output: str, *, job_id: str) -> SlurmJob | None:
    """`job_id`'s top-level `sacct` row (steps like `<id>.batch` ignored), None once vanished."""
    for line in output.splitlines():
        parts = [part.strip() for part in line.split("|")]
        if len(parts) >= 3 and parts[0] == job_id:
            return SlurmJob(
                job_id=job_id,
                state=parse_slurm_state(parts[1]),
                exit_code=parse_exit_code(parts[2]),
            )
    return None


def _build_resource_flags(resources: Resources) -> list[str]:
    """The allocation flags `sbatch` and `srun` both take, each only when set.

    A CPU-only job carries no `--gpus`, since a cluster without GPU GRES rejects one.
    """
    wanted = {
        f"--gpus={resources.gpus}": bool(resources.gpus),
        f"--time={resources.walltime}": resources.walltime is not None,
        f"--partition={resources.queue}": resources.queue is not None,
        f"--account={resources.account}": bool(resources.account),
        f"--mem={resources.mem_gb}G": resources.mem_gb is not None,
    }
    return [flag for flag, keep in wanted.items() if keep]


def build_sbatch_flags(resources: Resources, script: str) -> list[str]:
    """Render `resources` as `sbatch` flags, including the output sink and the script itself."""
    return ["sbatch", f"--output={_LOG_TEMPLATE}", *_build_resource_flags(resources), script]


def slurm_verdict(state: SlurmState | str | None, exit_code: int | None) -> str:
    """A one-word verdict for a SLURM job from its `sacct` state and exit code."""
    if state is None:
        return "vanished"
    if state in _SLURM_LIVE:
        return "running"
    if state == SlurmState.COMPLETED and (exit_code or 0) == 0:
        return "ok"
    return "failed"


class Slurm:
    """Dispatch jobs to a SLURM cluster via `sbatch`."""

    name = "slurm"

    def cancel(self, remote: Machine, root: str, *, handle: str) -> None:
        remote["bash"][["-lc", f"scancel {shlex.quote(handle)}"]](retcode=None)

    def interactive(self, *, env: str, command: Sequence[str], resources: Resources) -> str:
        """`srun --pty` under the batch flags; `env` is activated inside, beyond `srun`'s reach.

        Unlike PBS, `srun` runs `command` on the allocated node, and a login shell when empty.
        """
        flags = _build_resource_flags(resources)
        return shlex.join(["srun", "--pty", *flags, *(command or ("bash", "-l"))])

    def logs(self, remote: Machine, root: str, *, handle: str) -> str:
        return read_log(remote, root, handle=handle)

    def state(self, remote: Machine, root: str, *, handle: str) -> JobState:
        output = self.__cluster_command(remote, build_sacct_command(handle))
        if (job := parse_sacct_output(output, job_id=handle)) is None:
            return JobState(handle=handle, verdict=slurm_verdict(None, None))
        return JobState(
            handle=handle,
            state=str(job.state),
            exit_code=job.exit_code,
            verdict=slurm_verdict(job.state, job.exit_code),
        )

    def states(self, remote: Machine, root: str, handles: Sequence[str]) -> dict[str, JobState]:
        """Every job `squeue` still lists for this user, keyed by handle.

        A finished handle is left absent, since only `sacct` (the per-handle `state`) knows how it
        ended.
        """
        output = self.__cluster_command(remote, build_squeue_command())
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
        command = within(root, shlex.join(build_sbatch_flags(resources, script)))
        retcode, out, err = remote["bash"][["-lc", command]].run(retcode=None)
        handle = Slurm._extract_job_id(out)
        if not handle.isdigit():
            raise SystemExit(
                f"sbatch failed (rc={retcode}): {(err or out).strip()[-400:] or '(no output)'}"
            )
        return handle

    @staticmethod
    def _extract_job_id(output: str) -> str:
        """Pull the job id out of `sbatch` output (`Submitted batch job 12345`)."""
        if match := re.search(r"Submitted batch job\s+(\d+)", output):
            return match.group(1)
        return output.strip().splitlines()[-1].strip() if output.strip() else ""

    def __cluster_command(self, remote: Machine, command: list[str]) -> str:
        """Run a built `squeue`/`sacct`/`sinfo` argv under a login shell, returning its stdout."""
        return str(remote["bash"][["-lc", shlex.join(command)]](retcode=None))
