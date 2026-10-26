import subprocess
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from tempfile import TemporaryFile
from time import monotonic

from patos import FrozenModel
from pydantic import Field


class ToolOutput(FrozenModel):
    stdout: str = ""
    stderr: str = ""
    returncode: int = 0
    truncated: bool = False


class Runner(ABC):
    """Runs one external tool and keeps what it printed."""

    @abstractmethod
    def run(self, tool: str, args: Sequence[str], *, cwd: Path | None = None) -> ToolOutput:
        """`tool` with `args` in `cwd`, its output and exit status."""


class ToolRunner(Runner, FrozenModel):
    """Bound execution and retained output; never invoke a shell or run the artifact.

    deadline: the monotonic time past which no tool starts and a running one is stopped.
    """

    timeout: float = Field(default=120, gt=0)
    max_bytes: int = Field(default=32 << 20, gt=0)
    deadline: float | None = None

    def run(self, tool: str, args: Sequence[str], *, cwd: Path | None = None) -> ToolOutput:
        limit = self.timeout
        if self.deadline is not None:
            limit = min(limit, self.deadline - monotonic())
        if limit <= 0:
            return ToolOutput(stderr="tool deadline exhausted", returncode=124, truncated=True)
        with TemporaryFile() as stdout, TemporaryFile() as stderr:
            timed_out = False
            try:
                process = subprocess.run(
                    (tool, *args),
                    cwd=cwd,
                    stdout=stdout,
                    stderr=stderr,
                    timeout=limit,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                timed_out = True
                code = 124
            except OSError as fault:
                return ToolOutput(stderr=str(fault), returncode=127)
            else:
                code = process.returncode
            stdout.seek(0)
            stderr.seek(0)
            out, err = stdout.read(self.max_bytes + 1), stderr.read(self.max_bytes + 1)
        return ToolOutput(
            stdout=out[: self.max_bytes].decode("utf-8", errors="replace"),
            stderr=err[: self.max_bytes].decode("utf-8", errors="replace"),
            returncode=code,
            truncated=timed_out or len(out) > self.max_bytes or len(err) > self.max_bytes,
        )
