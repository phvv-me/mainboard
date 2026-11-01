# What a job's runner does with the output it shares with the command: where the streams go under
# PBS, the lines it adds in order with the command's own, and the exit status it keeps beside them.

import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Generator, Mapping


class JobOutput:
    """The output of one job's runner, in order with its command's.

    Every write flushes the runner's own streams first, since the command writes to the same
    descriptors and its output must land after what the runner said before it started.

    logs: the directory a PBS job appends its merged output and exit artifact to, empty for a
        backend that captures the output itself.
    environ: the environment the job runs in, which names the PBS job its log belongs to.
    """

    @contextmanager
    def redirected(self) -> Generator[None]:
        """Send a PBS job's merged output to the log a later poll reads, and nothing otherwise.

        A PBS server spools output where no poll looks and soon purges the job, so the job
        appends to its own log (exit status beside it). The runner's own standard streams move,
        as `exec >> log 2>&1` moved a script's, so everything lands in the one log until the run
        ends.
        """
        if not self.logs:
            yield
            return
        Path(self.logs).mkdir(parents=True, exist_ok=True)
        self.flush()
        kept = [os.dup(stream) for stream in (1, 2)]
        with (Path(self.logs) / f"{self.job_id}.log").open("ab") as log:
            for stream in (1, 2):
                os.dup2(log.fileno(), stream)
        try:
            yield
        finally:
            self._restore(kept)

    @property
    def job_id(self) -> str:
        """The PBS job's bare number, the stem its log and exit artifact are named by."""
        return self.environ["PBS_JOBID"].split(".", maxsplit=1)[0]

    def __init__(self, logs: str, environ: Mapping[str, str]) -> None:
        self.logs = logs
        self.environ = environ

    def _restore(self, kept: list[int]) -> None:
        """Point the runner's own streams back at the descriptors `redirected` saved in `kept`."""
        self.flush()
        for stream, descriptor in zip((1, 2), kept, strict=True):
            os.dup2(descriptor, stream)
            os.close(descriptor)

    def flush(self) -> None:
        """Flush this runner's own streams, so a child's output lands after what came before."""
        for stream in (sys.stdout, sys.stderr):
            stream.flush()

    def record_exit(self, status: int) -> None:
        """Say the exit status and keep it for a server that has since forgotten the job."""
        if self.logs:
            self.say(f"exit={status}")
            (Path(self.logs) / f"{self.job_id}.exit").write_text(
                f"exit={status}\n", encoding="utf-8", newline="\n"
            )

    def say(self, text: str, *, error: bool = False) -> None:
        """Write one line of the runner's own into the job's output, nothing for empty text.

        Straight to the descriptor like a shell's `echo`, so it lands wherever the command's
        output goes, a PBS log included, in order.
        """
        if text:
            self.flush()
            os.write(2 if error else 1, (text if text.endswith("\n") else f"{text}\n").encode())
