import subprocess
from pathlib import Path


def stub(path: Path, body: str) -> Path:
    """An executable `sh` script at `path` running `body`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"#!/bin/sh\n{body}\n", encoding="utf-8")
    path.chmod(0o755)
    return path


def login_environment(home: Path) -> dict[str, str]:
    """What a login under `home` sees: its own HOME and a PATH leading with the system stubs."""
    return {"HOME": str(home), "PATH": f"{home}/system-bin:/usr/bin:/bin"}


def sshd(home: Path, line: str, *, program: str) -> subprocess.CompletedProcess[str]:
    """Run `line` the way sshd does, through the login `program -c`, in `home`."""
    return subprocess.run(
        [program, "-c", line],
        env=login_environment(home),
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
