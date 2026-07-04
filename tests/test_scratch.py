from pathlib import Path

import pytest

import mainboard.models.scratch as scratch_mod
from mainboard.host import Host
from mainboard.models.scratch import Scratch

GIB = 1024**3


def clear_scratch_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Drop every scratch env var so only what a test sets is visible."""
    for key in scratch_mod.SCRATCH_ENV:
        monkeypatch.delenv(key, raising=False)


def test_env_var_wins_over_local_mounts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A scheduler `PBS_LOCALDIR` is chosen ahead of any bare local mount, with its free bytes."""
    clear_scratch_env(monkeypatch)
    monkeypatch.setenv("PBS_LOCALDIR", str(tmp_path))
    monkeypatch.setattr(
        scratch_mod.shutil, "disk_usage", lambda p: type("U", (), {"free": 500 * GIB})()
    )

    scratch = Scratch.probe()
    assert scratch.available is True
    assert scratch.path == tmp_path
    assert scratch.source == "PBS_LOCALDIR"
    assert scratch.free_bytes == 500 * GIB
    assert scratch.free_gb == 500.0


def test_falls_back_to_first_writable_literal_mount(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no env var set, the first existing writable literal mount is taken."""
    clear_scratch_env(monkeypatch)
    local = tmp_path / "local"
    local.mkdir()
    monkeypatch.setattr(scratch_mod, "SCRATCH_DIRS", ("/nonexistent", str(local)))
    monkeypatch.setattr(
        scratch_mod.shutil, "disk_usage", lambda p: type("U", (), {"free": 2 * GIB})()
    )

    scratch = Scratch.probe()
    assert scratch.path == local
    assert scratch.source == str(local)


def test_unwritable_candidate_is_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A candidate that exists but is not writable is passed over."""
    clear_scratch_env(monkeypatch)
    monkeypatch.setenv("LOCALDIR", str(tmp_path))
    monkeypatch.setattr(scratch_mod.os, "access", lambda path, mode: False)
    monkeypatch.setattr(scratch_mod, "SCRATCH_DIRS", ())

    scratch = Scratch.probe()
    assert scratch.available is False
    assert scratch.path is None
    assert scratch.free_bytes == 0


def test_no_writable_tier_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """When nothing is writable the tier is unavailable rather than raising."""
    clear_scratch_env(monkeypatch)
    monkeypatch.setattr(scratch_mod, "SCRATCH_DIRS", ("/nonexistent",))
    assert Scratch.probe().available is False


def test_host_exposes_scratch(monkeypatch: pytest.MonkeyPatch) -> None:
    """`Host.scratch` defers to the model probe."""
    sentinel = Scratch(path=Path("/local"), free_bytes=GIB, source="LOCALDIR")
    monkeypatch.setattr(Scratch, "probe", classmethod(lambda cls: sentinel))
    assert Host().scratch is sentinel
