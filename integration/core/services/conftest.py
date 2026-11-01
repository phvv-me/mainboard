from pathlib import Path

import pytest

from .fakes import Scripted


@pytest.fixture
def arming() -> str:
    """The command the settler's messages and fixes name for arming a stopped pass."""
    return "mb job monitor start"


@pytest.fixture
def manager() -> Scripted:
    """The service managers of a machine holding nothing yet."""
    return Scripted()


@pytest.fixture
def tool(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An installed `mainboard` as the only program on PATH."""
    script = tmp_path / "bin" / "mainboard"
    script.parent.mkdir()
    script.write_text("#!/bin/sh\n", encoding="utf-8")
    script.chmod(0o755)
    monkeypatch.setenv("PATH", str(script.parent))
    return script
