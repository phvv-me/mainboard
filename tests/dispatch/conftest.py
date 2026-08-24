from pathlib import Path

import pytest


@pytest.fixture
def workdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Run the body in a fresh empty CWD so cache/state files stay hermetic."""
    monkeypatch.chdir(tmp_path)
    return tmp_path
