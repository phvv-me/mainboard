import subprocess
import sys
from enum import StrEnum, auto
from os import PathLike
from pathlib import Path

from pydantic import Field, model_validator

from ..models.base import FrozenModel


class PythonAction(StrEnum):
    """A supported Tachyon operation."""

    RUN = auto()
    ATTACH = auto()
    DUMP = auto()
    REPLAY = auto()


class PythonMode(StrEnum):
    """The process state retained by Tachyon while sampling."""

    WALL = auto()
    CPU = auto()
    GIL = auto()
    EXCEPTION = auto()


class PythonFormat(StrEnum):
    """A Tachyon profile output representation."""

    PSTATS = auto()
    COLLAPSED = auto()
    FLAMEGRAPH = auto()
    GECKO = auto()
    HEATMAP = auto()
    BINARY = auto()


class AsyncMode(StrEnum):
    """The asyncio tasks included in an async-aware profile."""

    RUNNING = auto()
    ALL = auto()


class PythonProfile(FrozenModel):
    """One completed Tachyon command and its produced artifact or text."""

    action: PythonAction
    command: tuple[str, ...]
    target: str | None = None
    mode: PythonMode | None = None
    format: PythonFormat | None = None
    output: Path | None = None
    stdout: str = ""
    stderr: str = ""

    def __str__(self) -> str:
        return self.stdout or (str(self.output) if self.output is not None else "")


class Tachyon(FrozenModel):
    """Typed command interface for Python 3.15's external sampling profiler.

    executable: Python executable that runs Tachyon and the target. Attach operations
        require the exact same prerelease or minor Python version as the target process.
    mode: whether to retain wall, CPU, GIL, or exception samples.
    format: profile artifact representation.
    sampling_rate: samples per second as a number or a value such as `10khz`.
    duration: bounded profiling time in seconds. Omission profiles until target exit.
    all_threads: include every thread instead of only the main thread.
    native: mark transitions into native code.
    include_gc: retain synthetic garbage collection frames.
    async_aware: reconstruct logical asyncio task stacks across awaits.
    async_mode: include only running tasks or every waiting task.
    opcodes: collect adaptive bytecode instruction details.
    subprocesses: recursively profile Python child processes.
    blocking: pause the target while each consistent stack sample is read.
    output: output path. Omission prints pstats or lets Tachyon choose an artifact name.
    """

    executable: Path = Field(default_factory=lambda: Path(sys.executable))
    mode: PythonMode = PythonMode.WALL
    format: PythonFormat = PythonFormat.PSTATS
    sampling_rate: str = "1khz"
    duration: float | None = Field(default=None, gt=0)
    all_threads: bool = False
    native: bool = False
    include_gc: bool = True
    async_aware: bool = False
    async_mode: AsyncMode = AsyncMode.RUNNING
    opcodes: bool = False
    subprocesses: bool = False
    blocking: bool = False
    output: Path | None = None

    @model_validator(mode="after")
    def validate_options(self) -> Tachyon:
        """Reject combinations Tachyon cannot represent before starting a process."""
        if self.async_aware and (
            self.native or self.all_threads or self.mode in (PythonMode.CPU, PythonMode.GIL)
        ):
            raise ValueError(
                "async-aware profiling cannot combine with native frames, all threads, "
                "CPU mode, or GIL mode"
            )
        if self.opcodes and self.format not in (
            PythonFormat.FLAMEGRAPH,
            PythonFormat.GECKO,
            PythonFormat.HEATMAP,
        ):
            raise ValueError("opcode profiling requires flamegraph, gecko, or heatmap output")
        return self

    def run(
        self,
        target: str | PathLike[str],
        *,
        module: bool | None = None,
        args: tuple[str, ...] = (),
        timeout: float | None = None,
    ) -> PythonProfile:
        """Launch and sample a Python module or script from process startup.

        target: importable module name or script path.
        module: force module or script interpretation. Omission infers scripts from `.py`.
        args: arguments passed unchanged to the target.
        timeout: hard subprocess deadline in seconds.
        """
        target_text = str(target)
        is_module = not target_text.endswith(".py") if module is None else module
        command = [*self.command(PythonAction.RUN), *(("-m",) if is_module else ()), target_text]
        return self.execute(PythonAction.RUN, (*command, *args), timeout).model_copy(
            update={"target": target_text}
        )

    def attach(self, pid: int, *, timeout: float | None = None) -> PythonProfile:
        """Sample an already-running process with the matching Python executable."""
        command = (*self.command(PythonAction.ATTACH), str(pid))
        return self.execute(PythonAction.ATTACH, command, timeout)

    def dump(self, pid: int, *, timeout: float | None = None) -> PythonProfile:
        """Read one stack snapshot from a running Python process."""
        options = [str(self.executable), "-m", "profiling.sampling", PythonAction.DUMP]
        options.extend(self.dump_options())
        return self.execute(PythonAction.DUMP, (*options, str(pid)), timeout)

    def replay(
        self, profile: str | PathLike[str], *, timeout: float | None = None
    ) -> PythonProfile:
        """Render a recorded binary profile without sampling the target again."""
        command = [str(self.executable), "-m", "profiling.sampling", PythonAction.REPLAY]
        command.extend(self.output_options())
        return self.execute(PythonAction.REPLAY, (*command, str(profile)), timeout)

    def command(self, action: PythonAction) -> tuple[str, ...]:
        """Build the Tachyon command prefix and sampling options for `action`."""
        options = [str(self.executable), "-m", "profiling.sampling", action]
        options.extend(("--sampling-rate", self.sampling_rate, f"--mode={self.mode}"))
        if self.duration is not None:
            options.extend(("--duration", str(self.duration)))
        if self.all_threads:
            options.append("--all-threads")
        if self.native:
            options.append("--native")
        if not self.include_gc:
            options.append("--no-gc")
        if self.async_aware:
            options.extend(("--async-aware", f"--async-mode={self.async_mode}"))
        if self.opcodes:
            options.append("--opcodes")
        if self.subprocesses:
            options.append("--subprocesses")
        if self.blocking:
            options.append("--blocking")
        options.extend(self.output_options())
        return tuple(options)

    def output_options(self) -> list[str]:
        """Build the selected output format and optional destination arguments."""
        options = [f"--{self.format}"]
        if self.output is not None:
            options.extend(("--output", str(self.output)))
        return options

    def dump_options(self) -> list[str]:
        """Build the subset of options accepted by Tachyon's one-shot dump command."""
        options = []
        if self.all_threads:
            options.append("--all-threads")
        if self.native:
            options.append("--native")
        if not self.include_gc:
            options.append("--no-gc")
        if self.async_aware:
            options.extend(("--async-aware", f"--async-mode={self.async_mode}"))
        if self.opcodes:
            options.append("--opcodes")
        if self.blocking:
            options.append("--blocking")
        return options

    def execute(
        self, action: PythonAction, command: tuple[str, ...], timeout: float | None
    ) -> PythonProfile:
        """Run one bounded Tachyon command and preserve its text and artifact location."""
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return PythonProfile(
            action=action,
            command=command,
            mode=self.mode if action in (PythonAction.RUN, PythonAction.ATTACH) else None,
            format=self.format if action is not PythonAction.DUMP else None,
            output=self.output,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )

    def available(self) -> bool:
        """Whether `executable` provides the Python 3.15 sampling profiler."""
        try:
            subprocess.run(
                (str(self.executable), "-c", "import profiling.sampling"),
                check=True,
                capture_output=True,
                timeout=5.0,
            )
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return False
        return True
