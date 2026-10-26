# The PBS backend: submit through `qsub`, monitor through `qstat`, autopsy unresolved handles.

import re
import shlex
from collections.abc import Mapping
from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING

from patos import Model

from ....core.errors import MissionError
from ....core.project import Project
from ...jobs import vocabulary
from ...jobs.vocabulary import JobState, Resources
from ...shared import state_dir
from ..base import Scheduler, bare, login_ask, login_run, within

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ...transport.core import Machine

# The exit artifact a PBS job's runner writes on the host: one `exit=N` line.
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


# The full words some servers print instead of the letter: each member's name, `BEGUN` for arrays.
_WORD_STATE_ALIASES = {state.name.removeprefix("ARRAY_"): state for state in PbsState}

# PBS terminal states: the job has left the run queue.
_PBS_FINISHED = {PbsState.FINISHED, PbsState.EXITING}

# PBS states in which the job has not started, the ones an estimated start time is about.
_PBS_PENDING = {PbsState.QUEUED, PbsState.HELD, PbsState.WAITING, PbsState.MOVED}

# What `qdel` prints for a job that already left the queue, which a stop has nothing to add to.
_QDEL_GONE = ("finished", "Unknown")


class JobInfo(Model):
    """One `qstat`-parsed PBS job record.

    estimated_start / comment: the server's answer to "when does this queued job run", empty when
    it reports neither, the ordinary case for a running job.
    """

    job_id: str
    name: str
    state: PbsState | str
    queue: str
    exit_status: int | None = None  # set only for a finished job, else None
    queued_at: str = ""
    started_at: str = ""
    estimated_start: str = ""
    comment: str = ""


def _instant(stamp: str) -> str:
    """A qstat ctime stamp (`Thu Sep  4 14:00:00 2026`) as an ISO-8601 instant, else verbatim.

    qstat prints the login node's wall clock without a zone, so it is read in this process's zone.
    """
    try:
        return datetime.strptime(stamp, "%a %b %d %H:%M:%S %Y").astimezone().isoformat()
    except ValueError:
        return stamp


def build_qsub_flags(resources: Resources, *, scripted: bool = False) -> list[str]:
    """Render `resources` as `qsub` flags overriding the script's own `#PBS` header.

    scripted: the job script's header already selects the chunk, cards and memory included
        (`JobSpec`), and a command-line `select` replaces that chunk whole, so its GPU request
        was lost whenever memory was asked for (B17); only a scriptless `qsub -I` selects here.
    """
    selected = not scripted and resources.mem_gb is None and resources.nodes > 1
    wanted = [
        (resources.queue is not None, ["-q", f"{resources.queue}"]),
        (resources.walltime is not None, ["-l", f"walltime={resources.walltime}"]),
        (bool(resources.account), ["-W", f"group_list={resources.account}"]),
        (
            not scripted and resources.mem_gb is not None,
            ["-l", f"select={resources.nodes}:mem={resources.mem_gb}gb"],
        ),
        (selected, ["-l", f"select={resources.nodes}"]),
    ]
    return [flag for needed, flags in wanted if needed for flag in flags]


def qdel(remote: Machine, handle: str) -> None:
    """Delete PBS job `handle`, returning only once `qdel` accepted it or the job was already gone.

    Any other refusal raises, so a caller never reports a cancel the scheduler did not take.
    """
    code, _, err = login_ask(remote, f"qdel {shlex.quote(handle)}")
    if code and not any(gone in err for gone in _QDEL_GONE):
        raise MissionError(f"qdel {handle} failed ({code}): {err.strip()}")


class Pbs(Scheduler):
    """Dispatch jobs to a PBS cluster via `qsub`."""

    name = "pbs"

    @staticmethod
    def __job_state(handle: str, job: JobInfo) -> JobState:
        pending = job.state in _PBS_PENDING
        running = job.state is PbsState.RUNNING
        return JobState(
            handle=handle,
            label=job.name or None,
            state=str(job.state),
            exit_code=job.exit_status,
            verdict=Pbs._pbs_verdict(str(job.state), job.exit_status),
            stage=vocabulary.Word.QUEUED
            if pending
            else vocabulary.Word.RUNNING
            if running
            else "",
            since=job.started_at if running else job.queued_at if pending else "",
            estimated_start=job.estimated_start if pending else "",
            # The comment is where PBS says which resource the queue is short of, an answer to
            # "why not yet" that only a job still waiting has.
            note=job.comment if pending else "",
        )

    @staticmethod
    def _attributes(lines: Sequence[str]) -> dict[str, str]:
        """One job's `key = value` attributes, each tab-led continuation glued onto its line."""
        attributes: dict[str, str] = {}
        field = ""
        for line in lines:
            if line.startswith("\t") and field:
                attributes[field] += line.removeprefix("\t")
            elif " = " in line:
                key, _, value = line.partition(" = ")
                field = key.strip()
                attributes[field] = value
        return attributes

    @staticmethod
    def _extract_job_id(output: str) -> str:
        """The PBS job identifier from raw `qsub` output."""
        if match := re.match(r"^(\d+(?:\[[^\]]*\])?)\.?.*$", output.strip()):
            return match.group(1)
        return output.strip()

    @staticmethod
    def _job_info(job_id: str, attributes: Mapping[str, str]) -> JobInfo:
        """One parsed job from its `qstat -f` attribute block."""
        exit_status = attributes.get("Exit_status")
        return JobInfo(
            job_id=job_id,
            name=attributes.get("Job_Name", ""),
            state=Pbs._parsed_state(attributes.get("job_state", PbsState.QUEUED)),
            queue=attributes.get("queue", ""),
            exit_status=int(exit_status) if exit_status is not None else None,
            queued_at=_instant(attributes.get("qtime", "")),
            started_at=_instant(attributes.get("stime", "")),
            estimated_start=_instant(attributes.get("estimated.start_time", "")),
            comment=attributes.get("comment", ""),
        )

    @staticmethod
    def _parse_qstat_full(output: str) -> list[JobInfo]:
        """Parse `qstat -f` output into one record per job.

        qstat wraps a long attribute (in practice `comment`, the one saying why a queued job has
        not started) onto tab-led continuation lines broken mid-token, so each is glued back on
        with nothing between. An empty value is kept as empty, since the key says it was reported.
        """
        blocks = [
            block.splitlines()
            for block in re.split(r"^(?=Job Id:)", output, flags=re.MULTILINE)
            if block.startswith("Job Id:")
        ]
        return [
            Pbs._job_info(header.split(":", maxsplit=1)[1].strip(), Pbs._attributes(lines))
            for header, *lines in blocks
        ]

    @staticmethod
    def _parsed_state(value: str) -> PbsState | str:
        """A PBS job-state token (single letter or full word) as a state."""
        try:
            return PbsState(value)
        except ValueError:
            return _WORD_STATE_ALIASES.get(value.upper(), value)

    @staticmethod
    def _pbs_verdict(state: str | None, exit_code: int | None) -> str:
        """A one-word verdict for a PBS job from its state and exit status.

        A finished job with no exit status (e.g. `qdel`'d while still queued) is `unknown`, never
        `ok`, so a wait cannot report success for a job that produced nothing.
        """
        if state is None:
            return "vanished"
        if state not in _PBS_FINISHED:
            return "running"
        if exit_code is None:
            return "unknown"
        return "ok" if exit_code == 0 else "failed"

    def __query(
        self, remote: Machine, command: str, handles: Sequence[str]
    ) -> dict[str, JobState]:
        """One batched full-record qstat, keyed back to the requested handles by bare job id."""
        output = login_run(remote, f"{command} " + " ".join(shlex.quote(h) for h in handles))
        records = {bare(job.job_id): job for job in Pbs._parse_qstat_full(output)}
        return {
            handle: self.__job_state(handle, job)
            for handle in handles
            if (job := records.get(bare(handle))) is not None
        }

    def autopsy(self, remote: Machine, root: str, *, handle: str) -> JobState:
        """Settle a handle the server purged from the exit artifact its runner left on the host.

        `{STATE_DIR}/logs/<bare id>.exit` still yields a real `ok`/`failed`; no artifact (a
        hand-written script, a SIGKILL before the write) means the job is genuinely `vanished`.
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
        return JobState(handle=handle, verdict="vanished")

    def cancel(self, remote: Machine, root: str, *, handle: str) -> None:
        qdel(remote, handle)

    def interactive(self, *, env: str, command: Sequence[str], resources: Resources) -> str:
        """`qsub -I` under the batch flags; `env` is activated inside, beyond `qsub`'s reach.

        PBS hands over a login shell on the allocated node and takes no command, so a command is
        refused rather than quietly run on the login node.
        """
        if command:
            raise MissionError(
                "a PBS interactive session hands over a terminal and runs no command of its "
                f"own. Run `{Project().name} job submit` to dispatch one as a job."
            )
        return shlex.join(["qsub", "-I", *build_qsub_flags(resources)])

    def state(self, remote: Machine, root: str, *, handle: str) -> JobState:
        found = self.states(remote, root, [handle]).get(handle)
        return found if found is not None else self.autopsy(remote, root, handle=handle)

    def states(self, remote: Machine, root: str, handles: Sequence[str]) -> dict[str, JobState]:
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
        resources: Resources,
    ) -> str:
        flags = build_qsub_flags(resources, scripted=True)
        command = within(root, command=shlex.join(["qsub", *flags, script]))
        retcode, out, err = remote["bash"][["-lc", command]].run(retcode=None)
        handle = out.strip().splitlines()[-1] if out.strip() else ""
        if not handle[:1].isdigit():
            raise SystemExit(f"qsub failed (rc={retcode}): {(err or out).strip()[-400:]}")
        return Pbs._extract_job_id(handle)
