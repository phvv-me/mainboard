"""The real fleet, reached through `mb` from a real workspace: opt in with MB_REMOTE_WORKSPACE.

`MB_REMOTE_WORKSPACE=D:/projects pytest integration/dispatch/test_remote.py` checks that every
declared ssh host answers a command, which no fixture can stand in for.
"""

import os
import tomllib
from pathlib import Path

import pytest

_ROOT = os.environ.get("MB_REMOTE_WORKSPACE", "")
pytestmark = pytest.mark.skipif(
    not _ROOT, reason="set MB_REMOTE_WORKSPACE to reach the real fleet"
)


def hosts() -> list[str]:
    """Every declared ssh host of the workspace, empty when none is set."""
    if not _ROOT:
        return []
    root = Path(_ROOT)
    manifest = next(
        root / name for name in ("mb.toml", "mainboard.toml") if (root / name).is_file()
    )
    declared = tomllib.loads(manifest.read_text(encoding="utf-8")).get("hosts", {})
    return sorted(alias for alias, host in declared.items() if host.get("kind") == "ssh")


def test_the_fleet_is_listed_without_error(mb) -> None:
    assert mb("host", "list", cwd=Path(_ROOT), timeout=300).code == 0


@pytest.mark.parametrize("host", hosts())
def test_each_host_runs_a_command(mb, host: str) -> None:
    ran = mb("shell", "--on", host, "--", "hostname", cwd=Path(_ROOT), timeout=120)
    assert ran.code == 0, ran.said
