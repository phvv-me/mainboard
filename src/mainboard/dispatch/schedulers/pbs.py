# The PBS backend: submit through `qsub`, monitor through `qstat`, autopsy unresolved handles.

import re
import shlex
from enum import StrEnum
from typing import TYPE_CHECKING

from patos import Model

from ...core.errors import MissionError
from ...core.project import Project
from ..shared import state_dir
from ..vocabulary import JobState, Resources
from .base import login_run, read_log

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..transport import Machine

# The exit artifact the generated PBS job script traps out on the host: one `exit=N` line.
_EXIT_ARTIFACT = re.compile(r"exit=(\d+)")


class PbsState(StrEnum):
    """PBS job states, the single-letter codes `qstat` reports."""

    ARRAY_BEGUN = "B"
    EXITING = "E"
    FINISHED = "F"
    HELD = "H"
    MOVED = "M"
    QUEUED = "Q"
    RUNNING = "R"
    SUSPENDED = "S"
    WAITING = "W"


_WORD_STATE_ALIASES: dict[str, PbsState] = {
    "RUNNING": PbsState.RUNNING,
    "QUEUED": PbsState.QUEUED,
    "WAITING": PbsState.WAITING,
    "HELD": PbsState.HELD,
    "EXITING": PbsState.EXITING,
    "FINISHED": PbsState.FINISHED,
    "MOVED": PbsState.MOVED,
    "SUSPENDED": PbsState.SUSPENDED,
    "BEGUN": PbsState.ARRAY_BEGUN,
}

# PBS terminal states: the job has left the run queue.
_PBS_FINISHED = {PbsState.FINISHED, PbsState.EXITING}


def parse_job_state(value: str) -> PbsState | str:
    """Parse a PBS job-state token (single letter or full word)."""
    try:
        return PbsState(value)
    except ValueError:
        return _WORD_STATE_ALIASES.get(value.upper(), value)


class JobInfo(Model):
    """One `qstat`-parsed PBS job record."""

    job_id: str
    name: str
    state: PbsState | str
    queue: str
    exit_status: int | None = None  # set only for a finished job, else None


def parse_qstat_full(output: str) -> list[JobInfo]:
    """Parse `qstat -f` output."""
    jobs: list[JobInfo] = []
    current: JobInfo | None = None
    for line in output.splitlines():
        if line.startswith("Job Id:"):
            if current is not None:
                jobs.append(current)
            current = JobInfo(
                job_id=line.split(":", maxsplit=1)[1].strip(),
                name="",
                state=PbsState.QUEUED,
                queue="",
            )
            continue
        if current is None or " = " not in line:
            continue
        key, value = line.strip().split(" = ", maxsplit=1)
        match key:
            case "Job_Name":
                current.name = value
            case "job_state":
                current.state = parse_job_state(value)
            case "Exit_status":
                current.exit_status = int(value)
            case "queue":
                current.queue = value
    if current is not None:
        jobs.append(current)
    return jobs


def bare(handle: str) -> str:
    """A PBS handle's bare job number: `2435326.opbs` and `2435326` both -> `2435326`.

    The cache records what `qsub` printed (bare on some wrappers, `<id>.<server>` elsewhere)
    while `qstat -f` always reports the full id, so every lookup joins on the bare number rather
    than trusting the two spellings to agree.
    """
    return handle.split(".", maxsplit=1)[0]


def build_qsub_flags(resources: Resources) -> list[str]:
    """Render `resources` as `qsub` flags overriding the script's own `#PBS` header."""
    flags: list[str] = []
    if resources.queue is not None:
        flags += ["-q", resources.queue]
    if resources.walltime is not None:
        flags += ["-l", f"walltime={resources.walltime}"]
    if resources.account:
        flags += ["-W", f"group_list={resources.account}"]
    if resources.mem_gb is not None:
        flags += ["-l", f"select={resources.nodes}:mem={resources.mem_gb}gb"]
    return flags


class Pbs:
    """Dispatch jobs to a PBS cluster via `qsub`."""

    name = "pbs"

    def autopsy(self, remote: Machine, root: str, *, handle: str) -> JobState:
        """Settle a handle the scheduler no longer remembers from its on-host exit artifact.

        The generated PBS job script traps its exit into `{STATE_DIR}/logs/<bare jobid>.exit`,
        so a job that finished after the server purged its history still reconciles to a real
        `ok`/`failed` with its exit code. No artifact (a hand-written script, a SIGKILL that ran
        no trap) means the job is genuinely `vanished`.
        """
        artifact = shlex.quote(f"{root}/{state_dir()}/logs/{bare(handle)}.exit")
        out = login_run(remote, f"cat {artifact} 2>/dev/null")
        if match := _EXIT_ARTIFACT.search(out):
            code = int(match.group(1))
            return JobState(
                handle=handle,
                state="artifact",
                exit_code=code,
                verdict="ok" if code == 0 else "failed",
            )
        return JobState(handle=handle, state=None, exit_code=None, verdict="vanished")

    def cancel(self, remote: Machine, root: str, *, handle: str) -> None:
        del root
        remote["bash"][["-lc", f"qdel {shlex.quote(handle)}"]](retcode=None)

    def interactive(self, *, env: str, command: Sequence[str], resources: Resources) -> str:
        """An interactive PBS allocation, `qsub -I` under the same flags a batch submit renders.

        PBS hands the terminal a login shell on the node it allocates and takes no command to
        run there, so a command asked for here is refused rather than quietly run on the login
        node the allocation was requested from. The environment is likewise activated from
        inside the session, since nothing this side of `qsub` runs on the allocated node.
        """
        del env
        if command:
            raise MissionError(
                "a PBS interactive session hands over a terminal and runs no command of its "
                f"own. Run `{Project().name} submit` to dispatch one as a job."
            )
        return shlex.join(["qsub", "-I", *build_qsub_flags(resources)])

    def logs(self, remote: Machine, root: str, *, handle: str) -> str:
        return read_log(remote, root, handle=handle)

    def state(self, remote: Machine, root: str, *, handle: str) -> JobState:
        found = self.states(remote, root, [handle]).get(handle)
        return found if found is not None else self.autopsy(remote, root, handle=handle)

    def states(self, remote: Machine, root: str, handles: list[str]) -> dict[str, JobState]:
        del root
        if not handles:
            return {}
        found = self.__query(remote, "qstat -f", handles)
        if missing := [h for h in handles if h not in found]:
            found |= self.__query(remote, "qstat -f -H", missing)
        return found

    def submit(
        self,
        remote: Machine,
        root: str,
        *,
        script: str,
        args: Sequence[str],
        resources: Resources,
    ) -> str:
        del args  # PBS scripts are self-contained; qsub takes no free-form positional args.
        flags = build_qsub_flags(resources)
        command = shlex.join(["qsub", *flags, script])
        retcode, out, err = remote["bash"][["-lc", command]].run(retcode=None)
        handle = out.strip().splitlines()[-1] if out.strip() else ""
        if not handle[:1].isdigit():
            raise SystemExit(f"qsub failed (rc={retcode}): {(err or out).strip()[-400:]}")
        return Pbs._extract_job_id(handle)

    @staticmethod
    def _extract_job_id(output: str) -> str:
        """The PBS job identifier from raw `qsub` output."""
        if match := re.match(r"^(\d+(?:\[[^\]]*\])?)\.?.*$", output.strip()):
            return match.group(1)
        return output.strip()

    @staticmethod
    def __job_state(handle: str, job: JobInfo) -> JobState:
        return JobState(
            handle=handle,
            label=job.name or None,
            state=str(job.state),
            exit_code=job.exit_status,
            verdict=pbs_verdict(str(job.state), job.exit_status),
        )

    def __query(
        self, remote: Machine, command: str, handles: Sequence[str]
    ) -> dict[str, JobState]:
        """One batched full-record qstat, keyed back to the requested handles by bare job id."""
        output = login_run(remote, f"{command} " + " ".join(shlex.quote(h) for h in handles))
        records = {bare(job.job_id): job for job in parse_qstat_full(output)}
        return {
            handle: self.__job_state(handle, job)
            for handle in handles
            if (job := records.get(bare(handle))) is not None
        }


def pbs_verdict(state: str | None, exit_code: int | None) -> str:
    """A one-word verdict for a PBS job from its state and exit status.

    A finished job with no exit status (e.g. `qdel`'d while still queued) is `unknown`, never
    `ok`, so a wait cannot report success for a job that produced nothing.
    """
    if state is None:
        return "vanished"
    if state not in {member.value for member in _PBS_FINISHED}:
        return "running"
    if exit_code is None:
        return "unknown"
    return "ok" if exit_code == 0 else "failed"
