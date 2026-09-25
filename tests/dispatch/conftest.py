from pathlib import Path

import pytest


@pytest.fixture
def workdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Run the body in a fresh empty CWD so cache/state files stay hermetic."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture(autouse=True)
def fixture_keys_are_unlocked(monkeypatch: pytest.MonkeyPatch) -> None:
    """`support.keypair` writes placeholder text, not a key `ssh-keygen` could read.

    Whether a real key needs a passphrase has its own test, which sets this again.
    """
    monkeypatch.setattr("mainboard.dispatch.rentals.unlocked", lambda private: True)
