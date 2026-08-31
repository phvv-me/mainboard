import os
import time
from pathlib import Path

import pytest

from mainboard.trials import lease as lease_module
from mainboard.trials.lease import Busy, CardLease

# A pid at the top of the signed 32-bit range, which no live process on this machine holds.
_DEAD_PID = 2**31 - 1


def test_a_free_root_grants_the_lease_naming_this_process(tmp_path: Path) -> None:
    """The ordinary case: nobody else is here, so the lease is just taken and recorded."""
    lease = CardLease.acquire(tmp_path)
    pid, opened = (tmp_path / ".card.lock").read_text(encoding="utf-8").split()
    assert int(pid) == os.getpid()
    assert float(opened) == pytest.approx(time.time(), abs=5)
    assert lease.path == tmp_path / ".card.lock"


def test_release_removes_the_lease_and_tolerates_it_being_gone_already(tmp_path: Path) -> None:
    lease = CardLease.acquire(tmp_path)
    lease.release()
    assert not lease.path.exists()
    lease.release()


def test_a_live_holder_inside_its_ttl_refuses_naming_the_pid_and_the_age(tmp_path: Path) -> None:
    """A crashed session must not wedge every session after it, but a live one must be honoured."""
    path = tmp_path / ".card.lock"
    path.write_text(f"{os.getpid()} {time.time() - 5}", encoding="utf-8")
    with pytest.raises(Busy, match=f"held by pid {os.getpid()}") as raised:
        CardLease.acquire(tmp_path)
    assert raised.value.pid == os.getpid()
    assert raised.value.age == pytest.approx(5, abs=2)
    assert path.read_text(encoding="utf-8").split()[0] == str(os.getpid())


def test_a_lease_past_its_ttl_is_reclaimed_even_with_a_live_pid_behind_it(tmp_path: Path) -> None:
    path = tmp_path / ".card.lock"
    path.write_text(f"{os.getpid()} {time.time() - 100}", encoding="utf-8")
    lease = CardLease.acquire(tmp_path, ttl=10)
    assert lease.path == path
    assert path.read_text(encoding="utf-8").split()[0] == str(os.getpid())


def test_a_lease_naming_a_dead_pid_is_reclaimed_regardless_of_its_age(tmp_path: Path) -> None:
    path = tmp_path / ".card.lock"
    path.write_text(f"{_DEAD_PID} {time.time()}", encoding="utf-8")
    CardLease.acquire(tmp_path)
    assert path.read_text(encoding="utf-8").split()[0] == str(os.getpid())


def test_a_lease_file_that_cannot_be_read_as_a_pid_and_a_time_is_reclaimed(tmp_path: Path) -> None:
    """A torn or hand-edited lease file is not evidence of a live holder either."""
    path = tmp_path / ".card.lock"
    path.write_text("garbage", encoding="utf-8")
    CardLease.acquire(tmp_path)
    assert path.read_text(encoding="utf-8").split()[0] == str(os.getpid())

    path.write_text("not-a-pid not-a-time", encoding="utf-8")
    CardLease.acquire(tmp_path)
    assert path.read_text(encoding="utf-8").split()[0] == str(os.getpid())


def test_alive_treats_a_permission_refusal_as_still_running(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pid this process cannot signal still exists; only `ProcessLookupError` says otherwise."""

    def refuse(pid: int, sig: int) -> None:
        raise PermissionError

    monkeypatch.setattr(os, "kill", refuse)
    assert lease_module._alive(1) is True
