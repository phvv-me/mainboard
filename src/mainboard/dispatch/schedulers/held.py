"""The scheduler of the line `host hold` keeps: jobs run one at a time inside kept allocations.

It is `Pbs` for everything an allocation already is (the `qstat` of the allocation, the runner's
log and `.exit` artifacts under the PBS job id the node gives each job) and replaces only how a
job reaches a node: through the spool the line's `serve` reads (`dispatch.spool`), since compute
nodes accept no connection. A job's result is its `.exit` artifact, durable across allocations;
nothing is `vanished` on a missing record, only on PBS saying the allocation that claimed it
ended with no exit written.
"""

import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from patos import FrozenModel

from ...core.errors import MissionError
from ...core.project import Project
from ...runtime.job import walltime_seconds
from .. import vocabulary
from ..spool import Claim, LineSpec, Remote, Submission, spool_path
from ..spool import handle as new_handle
from ..transport import HostUnreachable
from ..vocabulary import JobState, Resources
from .base import login_ask
from .pbs import Pbs

if TYPE_CHECKING:
    from collections.abc import Collection, Sequence

    from ..transport import Machine

# How long a cancel waits for an allocation to end a job it already owns.
_CANCEL_SECONDS = 120.0
_POLL_SECONDS = 2.0


class Facts(FrozenModel):
    """What the spool said about some handles, read in one command.

    now: the login node's clock, the only one a deadline is compared with.
    """

    now: float
    line: LineSpec | None
    stopped: bool
    exits: dict[str, int]
    claims: dict[str, Claim]
    queued: set[str]


class Held(Pbs):
    """Jobs for the line `host hold` keeps, routed by a queue declaring `scheduler = "held"`."""

    name = "held"

    def cancel(self, remote: Machine, root: str, *, handle: str) -> None:
        """Stop the job, answering only once it can no longer run.

        That is fenced before any allocation took it, or finished with its exit artifact
        written; an allocation ends a claimed one on its tombstone.
        """
        said = login_ask(remote, Remote(spool_path(root)).fence(handle))[1].strip()
        if said != "claimed":
            return
        deadline = time.monotonic() + _CANCEL_SECONDS
        while time.monotonic() < deadline:
            if self.states(remote, root, [handle])[handle].verdict != vocabulary.RUNNING:
                return
            time.sleep(_POLL_SECONDS)
        raise MissionError(
            f"{handle} is still running in its allocation after {_CANCEL_SECONDS:g}s; its "
            "tombstone is set, so it ends there. Ask again before relying on it being stopped"
        )

    def enqueue(
        self, remote: Machine, root: str, *, script: str, resources: Resources, label: str
    ) -> str:
        """Queue `script` for the line, answering its handle once the entry is durable.

        The entry is exclusive and its handle chosen here, so a lost reply is reconciled by
        asking whether it landed and writing the same entry again if not; it never forks into two
        jobs. The line must exist and outlast the job.

        label: the creation intent the dispatch registry already holds for it.
        """
        spool = Remote(spool_path(root))
        facts = self._facts(remote, spool, [])
        walltime = resources.walltime or ""
        if facts.line is None or facts.stopped:
            raise MissionError(
                f"no line is held on this host; open one with `{Project().name} host hold "
                "<host> --for 2h`. It takes the user's one interactive slot, so a shell on the "
                "host is refused while it is held"
            )
        ends = datetime.fromtimestamp(facts.line.deadline, UTC).isoformat(timespec="minutes")
        if facts.now + walltime_seconds(walltime) + facts.line.grace > facts.line.deadline:
            raise MissionError(f"the held line ends {ends}, before a {walltime} job could finish")
        entry = Submission(
            handle=new_handle(),
            label=label,
            script=script,
            cwd=root,
            walltime=walltime,
            submitted=datetime.now(UTC).isoformat(),
        )
        self._publish(remote, spool, entry)
        return entry.handle

    def interactive(self, *, env: str, command: Sequence[str], resources: Resources) -> str:
        raise MissionError(
            f"a held line runs submitted jobs; for a shell run `{Project().name} shell --on "
            "<host>` once the line is released"
        )

    def submit(
        self,
        remote: Machine,
        root: str,
        *,
        script: str,
        args: Sequence[str],
        resources: Resources,
    ) -> str:
        del args  # a staged job script is self-contained
        return self.enqueue(remote, root, script=script, resources=resources, label="")

    def states(self, remote: Machine, root: str, handles: Sequence[str]) -> dict[str, JobState]:
        """Every handle's state from its own spool files, one command for all of them.

        A claim with no exit artifact is in flight until PBS says its allocation ended; an
        allocation PBS cannot find leaves it in flight rather than lost.
        """
        if not handles:
            return {}
        facts = self._facts(remote, Remote(spool_path(root)), list(handles))
        owners = {claim.alloc for h, claim in facts.claims.items() if h not in facts.exits}
        asked = Pbs.states(self, remote, root, sorted(owners - {""}))
        ended = {alloc for alloc, state in asked.items() if state.verdict != vocabulary.RUNNING}
        return {h: self._judge(h, facts, ended) for h in handles}

    def _facts(self, remote: Machine, spool: Remote, handles: list[str]) -> Facts:
        """Parse the spool's answer about `handles`; a missing spool is a refusal, not silence."""
        code, out, err = login_ask(remote, spool.read(handles))
        if code:
            raise MissionError(f"the hold spool {spool.path} cannot be read: {err.strip()[-200:]}")
        now, line, stopped = 0.0, None, False
        exits: dict[str, int] = {}
        claims: dict[str, Claim] = {}
        queued: set[str] = set()
        for row in out.splitlines():
            tag, _, rest = row.partition(" ")
            name, _, body = rest.partition(" ")
            if tag == "T":
                now = float(rest)
            elif tag == "L":
                line = LineSpec.model_validate_json(rest)
            elif tag == "S":
                stopped = True
            elif tag == "E":
                exits[name] = int(body.removeprefix("exit="))
            elif tag == "C":
                claims[name] = Claim.model_validate_json(body)
            elif tag == "Q":
                queued.add(name)
        return Facts(
            now=now, line=line, stopped=stopped, exits=exits, claims=claims, queued=queued
        )

    def _publish(self, remote: Machine, spool: Remote, entry: Submission) -> None:
        """Write the inbox entry, confirming by name when the reply was lost."""
        put = spool.put("inbox", name=f"{entry.handle}.json")
        for _ in range(2):
            answer = self._tried(remote, put, stdin=entry.model_dump_json())
            if answer is None:
                if entry.handle in self._facts(remote, spool, [entry.handle]).queued:
                    return
            elif answer[0]:
                raise MissionError(f"could not queue {entry.handle}: {answer[1].strip()[-200:]}")
            else:
                return
        raise HostUnreachable(
            f"{entry.handle} may or may not be queued; its creation intent {entry.label} is "
            "kept, reconcile by that handle before submitting again"
        )

    @staticmethod
    def _tried(remote: Machine, body: str, *, stdin: str) -> tuple[int, str] | None:
        """The status and stderr of `body`, None when the reply never came back."""
        try:
            code, _, err = login_ask(remote, body, stdin=stdin)
        except HostUnreachable:
            return None
        return code, err

    @staticmethod
    def _judge(name: str, facts: Facts, ended: Collection[str]) -> JobState:
        """One handle's state from what the spool holds for it."""
        if (code := facts.exits.get(name)) is not None:
            return JobState(
                handle=name,
                state="artifact",
                exit_code=code,
                verdict=vocabulary.OK if code == 0 else vocabulary.FAILED,
            )
        if (claim := facts.claims.get(name)) is not None:
            if claim.cancelled:
                return JobState(
                    handle=name, state=vocabulary.CANCELLED, verdict=vocabulary.CANCELLED
                )
            if claim.alloc in ended:
                return JobState(
                    handle=name,
                    verdict=vocabulary.VANISHED,
                    note=f"allocation {claim.alloc} ended while it ran, with no exit recorded",
                )
            since = datetime.fromtimestamp(claim.at, UTC).isoformat() if claim.at else ""
            return JobState(
                handle=name,
                state="running",
                verdict=vocabulary.RUNNING,
                stage=vocabulary.RUNNING,
                since=since,
                note=f"on {claim.node}" if claim.node else "",
            )
        line = facts.line
        over = line is None or facts.stopped or facts.now > line.deadline + line.grace
        if over and name in facts.queued:
            return JobState(
                handle=name,
                state=vocabulary.CANCELLED,
                verdict=vocabulary.CANCELLED,
                note="its line ended before any allocation took it",
            )
        if name not in facts.queued:
            return JobState(handle=name, verdict=vocabulary.VANISHED, note="no spool record")
        return JobState(
            handle=name,
            state="queued",
            verdict=vocabulary.RUNNING,
            stage=vocabulary.QUEUED,
            note="waiting for the held allocation",
        )
