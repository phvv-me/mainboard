import platform
import subprocess  # ruff:ignore[suspicious-subprocess-import]  reason=executes the fixed uv argv Mainboard generated, never a shell string
from collections.abc import Callable, Sequence
from pathlib import Path
from time import sleep

import psutil
from cyclopts import App

app = App()

# A launcher opened while uv replaces the Windows tool can briefly lock its ``Scripts``
# directory. Four short retries cover that handoff without turning a persistent failure into a
# long-running worker.
_WINDOWS_UV_RETRY_DELAYS = (0.25, 0.5, 1.0, 2.0)


def after_parent(
    parent: int,
    command: Sequence[str],
    log: Path,
    *,
    wait: Callable[[int], None] | None = None,
    execute: Callable[[Sequence[str]], subprocess.CompletedProcess[str]] | None = None,
    pause: Callable[[float], None] | None = None,
) -> int:
    """Wait for the running launcher to unlock, then replace its uv tool snapshot.

    parent: process holding the Windows launcher open.
    command: exact uv install argv generated from the installed receipt.
    log: durable stdout/stderr record beside the source workspace.
    wait: injectable parent waiter for the unit test.
    execute: injectable process boundary for the unit test.
    pause: injectable retry delay for the unit test.
    """
    if wait is None:
        wait = _wait
    if execute is None:
        execute = _execute
    if pause is None:
        pause = sleep
    wait(parent)
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text("", encoding="utf-8")
    try:
        return _installed(command, log, execute=execute, pause=pause)
    finally:
        # The marker the scheduling process left says an update is on its way; this one is done
        # with it however it ended, so the next stale command may schedule another.
        log.with_suffix(".pending").unlink(missing_ok=True)


def _installed(
    command: Sequence[str],
    log: Path,
    *,
    execute: Callable[[Sequence[str]], subprocess.CompletedProcess[str]],
    pause: Callable[[float], None],
) -> int:
    """Run the install, backing off while Windows still holds the tool directory locked."""
    for attempt, delay in enumerate(_WINDOWS_UV_RETRY_DELAYS, start=1):
        result = execute(command)
        _record(log, attempt, result)
        if not _windows_uv_tool_lock(result):
            return result.returncode
        pause(delay)
    result = execute(command)
    _record(log, len(_WINDOWS_UV_RETRY_DELAYS) + 1, result)
    return result.returncode


def _record(log: Path, attempt: int, result: subprocess.CompletedProcess[str]) -> None:
    """Append one completed refresh attempt to the durable worker transcript."""
    transcript = f"attempt={attempt} exit={result.returncode}\n{result.stdout}{result.stderr}"
    with log.open("a", encoding="utf-8") as stream:
        stream.write(transcript)
        if not transcript.endswith("\n"):
            stream.write("\n")
        stream.flush()


def _windows_uv_tool_lock(result: subprocess.CompletedProcess[str]) -> bool:
    """Whether uv hit the transient Windows directory lock the worker can safely retry."""
    failure = f"{result.stdout}\n{result.stderr}".casefold()
    return (
        result.returncode != 0
        and platform.system() == "Windows"
        and "failed to remove directory" in failure
        and any(code in failure for code in ("os error 5", "os error 32"))
    )


def _wait(parent: int) -> None:
    """Wait until `parent` releases the launcher, then install whatever the wait came to.

    Every way this wait can end leads to the same place: the install is attempted. A parent that
    vanished released the launcher, a parent still alive after the minute is one this worker has
    waited long enough for, a pid the system has since handed to somebody else is not the parent
    at all, and a process this user may not wait on cannot be watched any longer than this.

    Only the first of those was caught. The other two escaped before the log was even created,
    so a deferred update died in silence minutes after its launcher had already exited 0, with
    nothing on disk to say it had. Whatever the wait came to, uv is asked for the install and its
    own retry ladder answers a directory that is genuinely still locked.
    """
    try:
        psutil.Process(parent).wait(timeout=60.0)
    except psutil.NoSuchProcess, psutil.TimeoutExpired, psutil.AccessDenied:
        return


def _execute(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    """Run the already-tokenized uv install while retaining its full diagnostic."""
    return subprocess.run(command, capture_output=True, check=False, text=True)


@app.default
def main(parent: int, log: Path, *command: str) -> int:
    """Perform one deferred self-update after its parent Mainboard process exits."""
    return after_parent(parent, command, log)


if __name__ == "__main__":  # pragma: no cover - console-script fallback
    app()
