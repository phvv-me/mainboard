import platform
import subprocess  # ruff:ignore[suspicious-subprocess-import]  reason=executes the fixed uv argv Mainboard generated, never a shell string
from collections.abc import Sequence
from pathlib import Path
from time import sleep

import psutil
from cyclopts import App

app = App()

# A launcher opened while uv replaces the Windows tool can briefly lock its ``Scripts``
# directory. Four short retries cover that handoff without turning a persistent failure into a
# long-running worker.
_WINDOWS_UV_RETRY_DELAYS = (0.25, 0.5, 1.0, 2.0)


def after_parent(parent: int, command: Sequence[str], log: Path) -> int:
    """Wait for the running launcher to unlock, then replace its uv tool snapshot.

    parent: process holding the Windows launcher open.
    command: exact uv install argv generated from the installed receipt.
    log: durable stdout/stderr record beside the source workspace.
    """
    _wait(parent)
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text("", encoding="utf-8")
    try:
        for attempt, delay in enumerate(_WINDOWS_UV_RETRY_DELAYS, start=1):
            result = _attempt(command, log, attempt)
            if not _windows_uv_tool_lock(result):
                return result.returncode
            sleep(delay)
        return _attempt(command, log, len(_WINDOWS_UV_RETRY_DELAYS) + 1).returncode
    finally:
        # The marker the scheduling process left says an update is on its way; this one is done
        # with it however it ended, so the next stale command may schedule another.
        log.with_suffix(".pending").unlink(missing_ok=True)


def _attempt(command: Sequence[str], log: Path, attempt: int) -> subprocess.CompletedProcess[str]:
    """Run the install once, appending its full diagnostic to the durable worker transcript."""
    result = subprocess.run(command, capture_output=True, check=False, text=True)
    transcript = f"attempt={attempt} exit={result.returncode}\n{result.stdout}{result.stderr}"
    with log.open("a", encoding="utf-8") as stream:
        stream.write(transcript if transcript.endswith("\n") else f"{transcript}\n")
    return result


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
    """Wait until `parent` releases the launcher; every way the wait ends leads to the install.

    A vanished parent released the launcher, one still alive after the minute has been waited on
    long enough, a recycled pid is not the parent at all, and a process this user may not wait on
    cannot be watched longer. Only the first used to be caught: the others escaped before the log
    existed, so a deferred update died in silence minutes after its launcher exited 0. uv's own
    retry ladder answers a directory that is genuinely still locked.
    """
    try:
        psutil.Process(parent).wait(timeout=60.0)
    except psutil.NoSuchProcess, psutil.TimeoutExpired, psutil.AccessDenied:
        return


@app.default
def main(parent: int, log: Path, *command: str) -> int:
    """Perform one deferred self-update after its parent Mainboard process exits."""
    return after_parent(parent, command, log)


if __name__ == "__main__":  # pragma: no cover - console-script fallback
    app()
