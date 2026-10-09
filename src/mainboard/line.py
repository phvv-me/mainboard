# `mainboard host hold` and `host release` on a PBS host: keep the user's one interactive slot as
# a line that runs submitted jobs (`job submit --on <host> --queue held`).
#
# A rental is held as an ssh alias; a PBS node accepts no ssh, so a line is held as a spool on the
# shared filesystem and a keeper on the login node. The keeper is a generated shell loop in a
# detached tmux session, in the manner of a rental's entrypoint (`dispatch.rentals.waiting`): it
# opens `qsub -I` allocations one after another, each running `host serve`, until the line is
# released or past its deadline. Nothing parses a terminal or types into one. The center keeps no
# record: the line is whatever its spool says, so any session, on any login node, finds it.

import secrets
import shlex
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from patos import FrozenModel

from .context.admission import admit
from .core.errors import MissionError
from .core.project import Project
from .dispatch.schedulers.base import login_ask
from .dispatch.schedulers.pbs import build_qsub_flags
from .dispatch.shared import announce
from .dispatch.spool import Beat, LineSpec, Remote, spool_path
from .dispatch.vocabulary import Resources
from .dispatch.wrapping import connection
from .holds import duration_seconds
from .runtime.job import walltime_seconds

if TYPE_CHECKING:
    from .board import Board
    from .dispatch.shared import Watcher
    from .dispatch.transport import Machine

# A keeper or allocation silent this long is gone; its beats come far more often.
_FRESH = 60.0
# How long `hold` waits for the first allocation, and `release` for a graceful end, before
# acting on what it sees.
_READY = 150.0
_ENDS = 30.0
_POLL = 3.0
# No allocation is opened for less than this much of the line.
_LEAST = 300


class LineStatus(FrozenModel):
    """Where a host's held line stands, as its spool says now.

    state: `absent` (none), `opening` (kept, no allocation answering yet), `ready`, `busy` (a job
        runs), `renewing` (between allocations), `unkept` (deadline ahead but no keeper beating),
        `stopping` (released, a job still finishing) or `ended`.
    """

    host: str
    state: str
    queue: str = ""
    deadline: str = ""
    alloc: str = ""
    node: str = ""
    generation: int = 0
    keeper: str = ""
    running: str = ""
    detail: str = ""


class Line:
    """The held line of one PBS host: open it, read it, release it."""

    def __init__(self, board: Board) -> None:
        """board: bound to the PBS host whose interactive slot the line keeps."""
        self.board = board
        self.session = f"{Project().package}-hold-{board.host}"

    @property
    def host(self) -> str:
        """The PBS host this line keeps."""
        return self.board.host

    @classmethod
    def of(cls, board: Board, host: str) -> Line | None:
        """The line of `host` when it is a declared PBS host, else None (a rental or unknown)."""
        profile = board.manifest.profiles().get(host)
        return cls(board.on(host)) if profile is not None and profile.kind == "pbs" else None

    def open(self, duration: str, *, walltime: str = "", watch: Watcher = announce) -> LineStatus:
        """Keep the interactive slot for `duration`, answering once an allocation is serving.

        Idempotent: a line already kept is reported, never doubled. A first allocation that does
        not come (the slot is another job's) closes the line again and says why, rather than
        queueing behind it unseen.

        walltime: how long each allocation lasts, the queue's ceiling when empty.
        """
        plan = self.board.plan(container="none")
        queue = plan.profile.defaults.interact_queue
        if not queue:
            raise MissionError(f"{self.host} declares no interactive queue to keep")
        policy = plan.profile.policy(queue)
        seconds = duration_seconds(duration)
        ceilings = [walltime_seconds(w) for w in (policy.max_walltime, walltime) if w]
        length = min(seconds, *ceilings)
        admit(plan.profile, queue=queue, walltime=_hms(length), mem_gb=policy.mem_ceiling_gb)
        spec = LineSpec(
            queue=queue,
            walltime=_hms(length),
            mem_gb=policy.mem_ceiling_gb,
            account=plan.profile.account,
            deadline=time.time() + seconds,
            created=time.time(),
            tool=Project().package,
        )
        spool = Remote(spool_path(self.board.remote_root()))
        with connection(self.host) as remote:
            if (seen := self._status(remote, spool)).state not in {"absent", "unkept", "ended"}:
                return seen.model_copy(update={"detail": "already held"})
            watch(f"opening the line on {self.host}")
            self._start(remote, spool, spec)
            deadline = time.monotonic() + _READY
            while time.monotonic() < deadline:
                time.sleep(_POLL)
                if (seen := self._status(remote, spool)).state in {"ready", "busy"}:
                    return seen
            pane = self._pane(remote)
            self._stop(remote, spool)
        raise MissionError(
            f"no allocation answered on {self.host} within {_READY:g}s, so the line was closed. "
            f"The keeper's terminal said:\n{pane}"
        )

    def release(self) -> LineStatus:
        """End the line and free its allocation, letting a running job finish.

        Nothing more is claimed and the unclaimed are cancelled; the finishing job's results
        stay retrievable.
        """
        spool = Remote(spool_path(self.board.remote_root()))
        with connection(self.host) as remote:
            return self._stop(remote, spool)

    def status(self) -> LineStatus:
        """The line as its spool says now."""
        with connection(self.host) as remote:
            return self._status(remote, Remote(spool_path(self.board.remote_root())))

    def _start(self, remote: Machine, spool: Remote, spec: LineSpec) -> None:
        """Publish the line, then the keeper that serves it, under a fresh ownership token.

        The keeper runs under a login bash: tmux starts a pane in the server's environment, which
        for a server begun elsewhere has no `qsub` on its PATH.
        """
        token = secrets.token_hex(8)
        plan = self.board.plan(container="none")
        serve = self.board.line(
            f"{spec.tool} host serve {shlex.quote(spool.path)} --gen @GEN@ --walltime @WALL@",
            container="none",
        )
        flags = build_qsub_flags(
            Resources(queue=spec.queue, mem_gb=spec.mem_gb, account=plan.profile.account)
        )
        keeper = keeper_script(spec, spool=spool.path, token=token, flags=flags, template=serve)
        self._write(remote, spool, "line.json", text=spec.model_dump_json())
        self._write(remote, spool, "keeper.token", text=token)
        self._write(remote, spool, "keeper.sh", text=keeper)
        login_ask(remote, f"cd {shlex.quote(spool.path)} && rm -f stop keeper.beat")
        keeping = shlex.quote(f"bash -l {spool.path}/keeper.sh")
        started = login_ask(
            remote,
            f"tmux kill-session -t {self.session} 2>/dev/null; "
            f"tmux new-session -d -s {self.session} {keeping}",
        )
        if started[0]:
            raise MissionError(f"could not start the keeper on {self.host}: {started[2].strip()}")

    @staticmethod
    def _write(remote: Machine, spool: Remote, name: str, *, text: str) -> None:
        code, _, err = login_ask(remote, spool.write(name), stdin=text)
        if code:
            raise MissionError(f"could not write {name} into {spool.path}: {err.strip()[-200:]}")

    def _stop(self, remote: Machine, spool: Remote) -> LineStatus:
        """Stop the line, then free the allocation unless a job is still running in it.

        The allocation gets a few seconds to end itself before it is `qdel`led.
        """
        code, _, err = login_ask(remote, spool.stop())
        if code:
            raise MissionError(f"{self.host} has no held line to release ({err.strip()[-200:]})")
        deadline = time.monotonic() + _ENDS
        status = self._status(remote, spool)
        while status.alloc and not status.running and time.monotonic() < deadline:
            time.sleep(_POLL)
            status = self._status(remote, spool)
        if status.alloc and not status.running:
            freed = login_ask(remote, f"qdel {shlex.quote(status.alloc)}")
            if freed[0] and "finished" not in freed[2] and "Unknown" not in freed[2]:
                raise MissionError(f"qdel {status.alloc} failed ({freed[0]}): {freed[2].strip()}")
            status = self._status(remote, spool)
        return status

    def _status(self, remote: Machine, spool: Remote) -> LineStatus:
        """Read the spool's own files and name the state they add up to."""
        code, out, err = login_ask(remote, spool.status())
        if code:
            return LineStatus(host=self.host, state="absent", detail=err.strip()[-200:])
        seen: dict[str, str] = {}
        for row in out.splitlines():
            tag, _, rest = row.partition(" ")
            seen[tag] = rest
        if "L" not in seen:
            return LineStatus(host=self.host, state="absent")
        line = LineSpec.model_validate_json(seen["L"])
        now = float(seen["T"])
        beat = Beat.model_validate_json(seen["beat"]) if "beat" in seen else None
        fresh = beat is not None and now - beat.at < _FRESH
        kept = "B" in seen and now - float(seen["B"]) < _FRESH
        running = beat.running if fresh and beat is not None and "ended" not in seen else ""
        if "S" in seen:
            state = "stopping" if running else "ended"
        elif now > line.deadline:
            state = "ended"
        elif not kept:
            state = "unkept"
        elif "ended" in seen or not fresh:
            state = "opening" if "alloc" not in seen else "renewing"
        else:
            state = "busy" if running else "ready"
        return LineStatus(
            host=self.host,
            state=state,
            queue=line.queue,
            deadline=datetime.fromtimestamp(line.deadline, UTC).isoformat(timespec="minutes"),
            alloc=seen.get("alloc", "") if "ended" not in seen else "",
            node=seen.get("node", ""),
            generation=int(seen.get("G", "0")),
            keeper=seen.get("K", ""),
            running=running,
        )

    def _pane(self, remote: Machine) -> str:
        """What the keeper's terminal last showed, for the refusal that explains a failed open."""
        said = login_ask(
            remote, f"tmux capture-pane -p -t {self.session} 2>&1 | grep -v '^$' | tail -8"
        )[1]
        return said.strip() or "(nothing)"


def _hms(seconds: float) -> str:
    """`seconds` as the `HH:MM:SS` a walltime is written in."""
    return time.strftime("%H:%M:%S", time.gmtime(seconds))


def keeper_script(
    spec: LineSpec, *, spool: str, token: str, flags: list[str], template: str
) -> str:
    """The bash loop the keeper runs: one `qsub -I` allocation after another.

    Each generation is claimed with `mkdir`, so two keepers can never open the same one, and the
    token in `keeper.token` fences an older keeper out once a newer one is started. A refused
    `qsub` (another interactive job holds the slot) leaves no generation and is retried.

    template: the staged `host serve` line, `@GEN@` and `@WALL@` filled in per allocation. It
        runs as `/bin/bash -lc`: PBS execs the command itself, with a bare PATH.
    """
    return f"""#!/bin/bash
# Keeps a held line: opens one interactive allocation after another until it is released, fenced
# out by a newer keeper, or past its deadline. Generated by `host hold`.
spool={shlex.quote(spool)}; token={shlex.quote(token)}; deadline={int(spec.deadline)}
cap={walltime_seconds(spec.walltime)}
template={shlex.quote(template)}
own() {{ [ "$(cat "$spool/keeper.token" 2>/dev/null)" = "$token" ]; }}
say() {{ echo "$(date +%s) $*" >> "$spool/keeper.log"; }}
cd "$spool" || exit 1
printf '{{"host":"%s","pid":%s}}' "$(hostname -f)" "$$" > keeper.json.$$
mv -f keeper.json.$$ keeper.json
( while own; do date +%s > keeper.beat.$$ && mv -f keeper.beat.$$ keeper.beat; sleep 15; done ) &
beater=$!
trap 'kill $beater 2>/dev/null' EXIT
mkdir -p gen
n=$(( $(ls gen | sort -n | tail -1) + 1 ))
while [ ! -e stop ] && own; do
  left=$(( deadline - $(date +%s) ))
  [ "$left" -gt {_LEAST} ] || break
  wall=$(( left < cap ? left : cap ))
  wall=$(printf '%02d:%02d:%02d' $((wall / 3600)) $((wall % 3600 / 60)) $((wall % 60)))
  if mkdir "gen/$n" 2>/dev/null; then
    line=${{template//@GEN@/$n}}; line=${{line//@WALL@/$wall}}
    say "generation $n for $wall"
    qsub -I {shlex.join(flags)} -l "walltime=$wall" -N mbhold -- /bin/bash -lc "$line"
    say "generation $n ended, qsub exit $?"
    if [ -e "gen/$n/alloc" ]; then n=$((n + 1)); else rmdir "gen/$n" 2>/dev/null; sleep 10; fi
  else
    n=$((n + 1))
  fi
done
say "keeper finished"
"""
