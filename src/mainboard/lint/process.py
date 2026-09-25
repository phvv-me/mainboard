import shutil
import subprocess
import time
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING

import psutil
from patos import FrozenModel

if TYPE_CHECKING:
    from collections.abc import Mapping

# The exit codes a command did not choose: the shell's words for "not found" and "timed out".
MISSING = 127
TIMED_OUT = 124


class Outcome(FrozenModel):
    """One lint step's verdict: which step, in which owner, how it exited and what it said.

    step: the tool name, or `text` for the built-in hygiene.
    owner: the owner directory, workspace-relative, `.` for the root.
    code: the exit code, `124` for a step killed at its deadline and `127` for a missing tool.
    output: everything the step printed, stdout and stderr interleaved.
    """

    step: str
    owner: str
    code: int
    seconds: float
    output: str

    @property
    def failed(self) -> bool:
        return self.code != 0


class Invocation(FrozenModel):
    """One command a lint tool runs in one owner directory.

    step: the tool name.
    owner: the owner directory, workspace-relative.
    cwd: the directory the command runs in, the owner's absolute path.
    argv: the expanded command line.
    env: the environment whose PATH resolves the command.
    timeout: seconds before the command and every process it started are killed.
    """

    step: str
    owner: str
    cwd: Path
    argv: tuple[str, ...]
    env: str
    timeout: float

    def run(self, environment: Mapping[str, str]) -> Outcome:
        """Run the command under `environment`, stopping it and its children at the deadline.

        The executable is looked up on the environment's own PATH, because Windows resolves a
        bare program name against the parent's PATH whatever the child is handed.

        environment: the complete process environment the command runs under.
        """
        started = time.monotonic()
        executable = shutil.which(self.argv[0], path=environment.get("PATH"))
        if executable is None:
            return self._outcome(
                MISSING, started, f"{self.argv[0]} is not on the {self.env} environment's PATH"
            )
        process = subprocess.Popen(
            [executable, *self.argv[1:]],
            cwd=self.cwd,
            env=dict(environment),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
        )
        try:
            output, _ = process.communicate(timeout=self.timeout)
        except subprocess.TimeoutExpired:
            _kill(process)
            output, _ = process.communicate()
            said = output.decode(errors="replace")
            return self._outcome(
                TIMED_OUT, started, f"{said}\n{self.step} exceeded its {self.timeout:g}s deadline"
            )
        return self._outcome(process.returncode, started, output.decode(errors="replace"))

    def _outcome(self, code: int, started: float, output: str) -> Outcome:
        return Outcome(
            step=self.step,
            owner=self.owner,
            code=code,
            seconds=time.monotonic() - started,
            output=output,
        )


def _kill(process: subprocess.Popen[bytes]) -> None:
    """Kill `process` and every descendant, which keep its output pipe open until they die."""
    for child in psutil.Process(process.pid).children(recursive=True):
        with suppress(psutil.NoSuchProcess):
            child.kill()
    process.kill()
