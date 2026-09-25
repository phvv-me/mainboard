import os
import time
from collections.abc import Callable
from pathlib import Path

import pytest

from mainboard.trials import lease as lease_module
from mainboard.trials.lease import Busy, CardLease

# A pid at the top of the signed 32-bit range, which no live process on this machine holds.
_DEAD_PID = 2**31 - 1


def holder(path: Path) -> str:
    return path.read_text(encoding="utf-8").split()[0]


def test_a_free_root_grants_the_lease_naming_this_process(tmp_path: Path) -> None:
    lease = CardLease.acquire(tmp_path)
    pid, opened = lease.path.read_text(encoding="utf-8").split()
    assert int(pid) == os.getpid()
    assert float(opened) == pytest.approx(time.time(), abs=5)
    assert lease.path == tmp_path / lease_module.filename()
    lease.release()
    assert not lease.path.exists()
    lease.release()


def test_a_live_holder_inside_its_ttl_refuses_naming_the_pid_and_the_age(tmp_path: Path) -> None:
    path = tmp_path / lease_module.filename()
    path.write_text(f"{os.getpid()} {time.time() - 5}", encoding="utf-8")
    with pytest.raises(Busy, match=f"held by pid {os.getpid()}") as raised:
        CardLease.acquire(tmp_path)
    assert raised.value.pid == os.getpid()
    assert raised.value.age == pytest.approx(5, abs=2)
    assert holder(path) == str(os.getpid())


@pytest.mark.parametrize(
    ("content", "ttl"),
    [
        (lambda: f"{os.getpid()} {time.time() - 100}", 10),
        (lambda: f"{_DEAD_PID} {time.time()}", lease_module.DEFAULT_TTL_S),
        (lambda: "garbage", lease_module.DEFAULT_TTL_S),
        (lambda: "not-a-pid not-a-time", lease_module.DEFAULT_TTL_S),
    ],
    ids=["past-ttl-live-pid", "dead-pid", "torn", "unparsable"],
)
def test_a_stale_or_unreadable_lease_is_reclaimed(
    tmp_path: Path, content: Callable[[], str], ttl: float
) -> None:
    path = tmp_path / lease_module.filename()
    path.write_text(content(), encoding="utf-8")
    assert CardLease.acquire(tmp_path, ttl=ttl).path == path
    assert holder(path) == str(os.getpid())


def test_two_hosts_sharing_one_root_hold_separate_leases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cluster's universe root is one filesystem for every node, so a lease names its host."""
    monkeypatch.setattr(lease_module.socket, "gethostname", lambda: "node-a")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    first = CardLease.acquire(tmp_path)
    monkeypatch.setattr(lease_module.socket, "gethostname", lambda: "node-b")
    second = CardLease.acquire(tmp_path)
    assert first.path.name == ".card.lock.node-a.0"
    assert second.path.name == ".card.lock.node-b.0"
    monkeypatch.setattr(lease_module.socket, "gethostname", lambda: "node-a")
    with pytest.raises(Busy):
        CardLease.acquire(tmp_path)
