"""`serve agents sync` writes the owner's own harness settings, so an agent is refused it."""

from pathlib import Path

import pytest


def test_serve_agents_sync_refuses_an_agent_and_writes_nothing(
    mb, workspace: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    home = tmp_path_factory.mktemp("home")
    ran = mb("serve", "agents", "sync", env={"MB_AGENT": "1", "HOME": str(home)})
    assert (ran.code, "`mb serve agents sync` is the owner's act" in ran.err) == (1, True)
    assert (list(home.iterdir()), (workspace / ".mb" / "serve").exists()) == ([], False)
