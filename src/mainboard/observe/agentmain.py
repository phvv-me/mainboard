# The tiny module a wrapped job actually runs on the node: `python -m mainboard.observe.agentmain
# --root R --job J -- child args...` spools the child's output, samples its memory, and records
# its exit; `--follow --from-offset N` instead replays and tails an already-running job's spool,
# printing NDJSON to stdout for a `StreamChannel` on the other end of the ssh exec to read.

import subprocess  # ruff:ignore[suspicious-subprocess-import]  reason=argv is the job wrapper's own command, not untrusted input since=2026-08-17
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import psutil

from .frames import Frame, Kind, encode
from .spool import Spool, follow

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from .frames import JSONValue


class Args(NamedTuple):
    """Parsed `agentmain` invocation: either a follow read or a wrap-and-run.

    root: the spool root every job directory lives under.
    job: the job identity this invocation observes.
    follow: replay-and-tail an existing spool instead of wrapping a child.
    from_offset: the checkpoint a follow replay starts at.
    child: the wrapped command and its arguments, everything after a literal `--`.
    """

    root: str
    job: str
    follow: bool
    from_offset: int
    child: tuple[str, ...]


def parse_args(argv: Sequence[str]) -> Args:
    """Parse `agentmain`'s CLI flags; everything after `--` becomes the wrapped child argv."""
    root = job = ""
    following = False
    from_offset = 0
    child: list[str] = []
    flags = iter(argv)
    for flag in flags:
        match flag:
            case "--root":
                root = next(flags)
            case "--job":
                job = next(flags)
            case "--follow":
                following = True
            case "--from-offset":
                from_offset = int(next(flags))
            case "--":
                child = list(flags)
    return Args(root=root, job=job, follow=following, from_offset=from_offset, child=tuple(child))


def main(argv: Sequence[str]) -> int:
    """`agentmain`'s entry point: replay-and-follow, or wrap-and-run, per the parsed flags."""

    args = parse_args(argv)
    with Spool(Path(args.root), args.job) as spool:
        if args.follow:
            for frame in follow(spool, args.from_offset):
                sys.stdout.write(encode(frame))
                sys.stdout.flush()
            return 0
        return wrap(spool, args.child)


def wrap(
    spool: Spool,
    argv: Sequence[str],
    *,
    sample_interval: float = 5.0,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
    rss: Callable[[int], int] = lambda pid: psutil.Process(pid).memory_info().rss,
) -> int:
    """Run `argv` as the wrapped child, spooling its output, periodic rss samples, and exit code.

    spool: where every frame and heartbeat for this job lands.
    argv: the child command to run and observe.
    sample_interval: minimum real seconds between rss samples, checked between output lines.
    now: overridden in a test for a deterministic clock.
    rss: overridden in a test to avoid depending on a real child process's memory.
    """
    process = subprocess.Popen(  # ruff:ignore[subprocess-without-shell-equals-true]  reason=argv is the job wrapper's own command, not untrusted input since=2026-08-17
        list(argv), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    assert process.stdout is not None  # ruff:ignore[assert]  reason=guaranteed by stdout=PIPE just above since=2026-08-17
    arguments: list[JSONValue] = [*argv]
    spool.append(_frame(spool.job, Kind.started, now(), {"argv": arguments}))
    sampled_at = now()
    for line in process.stdout:
        spool.append(_frame(spool.job, Kind.line, now(), {"text": line.rstrip("\r\n")}))
        moment = now()
        if (moment - sampled_at).total_seconds() >= sample_interval:
            spool.append(_frame(spool.job, Kind.sample, moment, {"rss": rss(process.pid)}))
            spool.heartbeat("running")
            sampled_at = moment
    code = process.wait()
    spool.append(_frame(spool.job, Kind.ended, now(), {"exit_code": code}))
    spool.heartbeat("ended")
    return code


def _frame(job: str, kind: Kind, at: datetime, payload: dict[str, JSONValue]) -> Frame:
    return Frame(job=job, kind=kind, at=at, payload=payload)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main(sys.argv[1:]))
