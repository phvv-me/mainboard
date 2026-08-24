from pathlib import Path

import pytest

from mainboard.probe import Scratch
from mainboard.probe.facts import scratch as scratch_mod

_GIB = 1024**3


@pytest.fixture
def bare_host(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    """A host with no scratch env var set, no local mount, and a fixed free-space reading."""
    for key in scratch_mod._SCRATCH_ENV:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(scratch_mod, "_SCRATCH_DIRS", ())
    monkeypatch.setattr(
        scratch_mod.shutil, "disk_usage", lambda path: type("Usage", (), {"free": 500 * _GIB})()
    )
    return monkeypatch


def test_a_scheduler_scratch_variable_is_taken_before_any_bare_local_mount(
    tmp_path: Path, bare_host: pytest.MonkeyPatch
) -> None:
    """A scheduler's own scratch variable beats every mount scan.

    PBS and SLURM hand a job its own node-local NVMe through an env var, and that beats the
    shared `/tmp` a bare mount scan would otherwise settle for.
    """
    bare_host.setenv("PBS_LOCALDIR", str(tmp_path))
    bare_host.setattr(scratch_mod, "_SCRATCH_DIRS", ("/nonexistent", str(tmp_path / "unused")))

    scratch = Scratch.probe()
    assert scratch.available is True
    assert scratch.path == tmp_path
    assert scratch.source == "PBS_LOCALDIR"
    assert scratch.free_bytes == 500 * _GIB
    assert scratch.free_gb == 500.0


def test_without_a_variable_the_first_existing_writable_mount_wins(
    tmp_path: Path, bare_host: pytest.MonkeyPatch
) -> None:
    """Mount candidates are tried in cluster-convention order.

    The ones that are not there are skipped, and the chosen path records which candidate it
    came from.
    """
    local = tmp_path / "local"
    local.mkdir()
    bare_host.setattr(scratch_mod, "_SCRATCH_DIRS", ("/nonexistent", str(local)))

    scratch = Scratch.probe()
    assert scratch.path == local
    assert scratch.source == str(local)


@pytest.mark.parametrize("writable", [True, False], ids=["no-candidate", "unwritable-candidate"])
def test_a_host_with_nothing_writable_reports_an_unavailable_tier(
    writable: bool, tmp_path: Path, bare_host: pytest.MonkeyPatch
) -> None:
    """An unwritable scratch candidate is passed over like a missing one.

    A caller has to be able to tell node-local NVMe from a shared filesystem.
    """
    if not writable:
        bare_host.setenv("LOCALDIR", str(tmp_path))
        bare_host.setattr(scratch_mod.os, "access", lambda path, mode: False)

    scratch = Scratch.probe()
    assert scratch.available is False
    assert scratch.path is None
    assert scratch.free_bytes == 0
