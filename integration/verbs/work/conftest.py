import os
from pathlib import Path

import pytest

from .constants import MANAGERS


@pytest.fixture
def managers(tmp_path: Path) -> dict[str, str]:
    """The environment of a command that finds only service managers with nothing installed.

    Their programs lead its PATH, and each notes what it was asked in the file FAKE_MANAGER_LOG
    names, which is how a test knows no real manager answered.
    """
    programs = tmp_path / "managers"
    programs.mkdir()
    for name, text in MANAGERS.items():
        (programs / name).write_text(text, encoding="utf-8")
        (programs / name).chmod(0o755)
    return {
        "PATH": f"{programs}{os.pathsep}{os.environ['PATH']}",
        "FAKE_MANAGER_LOG": str(tmp_path / "managers.log"),
    }
