import subprocess
from pathlib import Path
from tempfile import TemporaryFile
from time import monotonic
from typing import Protocol

from patos import FrozenModel


class ToolOutput(FrozenModel):
    stdout: str = ""
    stderr: str = ""
    returncode: int = 0
    truncated: bool = False


class Runner(Protocol):
    def run(self, tool: str, args: tuple[str, ...], *, cwd: Path | None = None) -> ToolOutput: ...


class ToolRunner:
    """Bound execution and retained output; never invoke a shell or run the artifact."""

    def __init__(
        self, *, timeout: float = 120, max_bytes: int = 32 << 20, deadline: float | None = None
    ) -> None:
        if timeout <= 0 or max_bytes <= 0:
            raise ValueError("tool timeout and output limit must be positive")
        self.timeout = timeout
        self.max_bytes = max_bytes
        self.deadline = deadline

    def run(self, tool: str, args: tuple[str, ...], *, cwd: Path | None = None) -> ToolOutput:
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
