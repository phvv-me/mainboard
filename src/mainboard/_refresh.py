import subprocess  # ruff:ignore[suspicious-subprocess-import]  reason=executes the fixed uv argv Mainboard generated, never a shell string
from collections.abc import Callable, Sequence
from pathlib import Path

import psutil
from cyclopts import App

app = App()


def after_parent(
    parent: int,
    command: Sequence[str],
    log: Path,
    *,
    wait: Callable[[int], None] | None = None,
    execute: Callable[[Sequence[str]], subprocess.CompletedProcess[str]] | None = None,
) -> int:
    """Wait for the running launcher to unlock, then replace its uv tool snapshot.

    parent: process holding the Windows launcher open.
    command: exact uv install argv generated from the installed receipt.
    log: durable stdout/stderr record beside the source workspace.
    wait: injectable parent waiter for the unit test.
    execute: injectable process boundary for the unit test.
    """
    if wait is None:
        wait = _wait
    if execute is None:
        execute = _execute
    wait(parent)
    result = execute(command)
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(f"exit={result.returncode}\n{result.stdout}{result.stderr}", encoding="utf-8")
    return result.returncode


def _wait(parent: int) -> None:
    """Wait until `parent` releases the launcher, already released when it vanished."""
    try:
        psutil.Process(parent).wait(timeout=60.0)
    except psutil.NoSuchProcess:
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
