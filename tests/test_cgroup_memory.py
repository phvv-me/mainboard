from pathlib import Path

import pytest

import mainboard.models.cgroup_memory as cgroup_mod
from mainboard.host import Host
from mainboard.models.cgroup_memory import CgroupMemory

GIB = 1024**3
UNLIMITED_V1 = "9223372036854771712"  # the kernel's page-aligned near-2**63 "no limit" sentinel


def point_cgroup_root(monkeypatch: pytest.MonkeyPatch, root: Path) -> None:
    """Redirect the model's `/sys/fs/cgroup` root at a tmp tree."""
    monkeypatch.setattr(cgroup_mod, "CGROUP_ROOT", root)


def point_proc(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, text: str) -> None:
    """Make the membership file read as the given text via a real tmp file."""
    proc = tmp_path / "self.cgroup"
    proc.write_text(text)
    monkeypatch.setattr(cgroup_mod, "CGROUP_PROC", proc)


def test_v2_ancestor_walk_takes_the_tightest_finite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """v2 caps a job on an ancestor while the leaf reads `max`; the ancestor cap wins."""
    point_cgroup_root(monkeypatch, tmp_path)
    job = tmp_path / "job.123"
    leaf = job / "task"
    leaf.mkdir(parents=True)
    (job / "memory.max").write_text(str(100 * GIB))  # the jobid scope carries the real cap
    (leaf / "memory.max").write_text("max")  # the process's own leaf is uncapped
    point_proc(monkeypatch, tmp_path, "0::/job.123/task\n")

    cgroup = CgroupMemory.probe()
    assert cgroup.capped is True
    assert cgroup.limit_bytes == 100 * GIB
    assert cgroup.limit_gb == 100.0


def test_v1_reads_memsw_on_an_ancestor_with_leaf_unset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """v1 (Miyabi GH200 PBS): memsw on an ancestor, the leaf left at the unlimited sentinel."""
    point_cgroup_root(monkeypatch, tmp_path)
    memory = tmp_path / "memory"
    job = memory / "pbspro" / "42.miyabi"
    leaf = job / "0"
    leaf.mkdir(parents=True)
    # the ancestor carries a generous RAM limit but a tighter memsw ceiling, the real OOM line
    (job / "memory.limit_in_bytes").write_text(str(120 * GIB))
    (job / "memory.memsw.limit_in_bytes").write_text(str(110 * GIB))
    (leaf / "memory.limit_in_bytes").write_text(UNLIMITED_V1)
    (leaf / "memory.memsw.limit_in_bytes").write_text(UNLIMITED_V1)
    point_proc(
        monkeypatch, tmp_path, "5:memory:/pbspro/42.miyabi/0\n4:cpu,cpuacct:/pbspro/42.miyabi/0\n"
    )

    cgroup = CgroupMemory.probe()
    assert cgroup.capped is True
    assert cgroup.limit_bytes == 110 * GIB  # memsw is tighter than the RAM limit


def test_uncapped_host_falls_back_to_total_ram(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A host with no finite cap (the bare 4090 box) reports total RAM and `capped=False`."""
    point_cgroup_root(monkeypatch, tmp_path)
    leaf = tmp_path / "user.slice"
    leaf.mkdir()
    (leaf / "memory.max").write_text("max")
    point_proc(monkeypatch, tmp_path, "0::/user.slice\n")
    monkeypatch.setattr(
        cgroup_mod.psutil, "virtual_memory", lambda: type("VM", (), {"total": 64 * GIB})()
    )

    cgroup = CgroupMemory.probe()
    assert cgroup.capped is False
    assert cgroup.limit_bytes == 64 * GIB


def test_unreadable_proc_cgroup_is_uncapped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unreadable `/proc/self/cgroup` (a non-Linux host) degrades to total RAM, never raises."""
    point_cgroup_root(monkeypatch, tmp_path)
    monkeypatch.setattr(cgroup_mod, "CGROUP_PROC", tmp_path / "absent")  # never created
    monkeypatch.setattr(
        cgroup_mod.psutil, "virtual_memory", lambda: type("VM", (), {"total": 8 * GIB})()
    )
    assert CgroupMemory.enforced_limit() is None
    assert CgroupMemory.probe().limit_bytes == 8 * GIB


def test_no_recognized_membership_line_is_uncapped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cgroup file with neither a v2 unified line nor a memory controller yields no cap."""
    point_cgroup_root(monkeypatch, tmp_path)
    point_proc(monkeypatch, tmp_path, "3:cpu,cpuacct:/some/path\n")
    assert CgroupMemory.enforced_limit() is None


def test_corrupt_cap_file_is_skipped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-integer cap file is ignored rather than raising; a sibling finite cap still wins."""
    point_cgroup_root(monkeypatch, tmp_path)
    node = tmp_path / "memory" / "scope"  # v1 nests the memory controller under <root>/memory
    node.mkdir(parents=True)
    (node / "memory.limit_in_bytes").write_text("garbage")
    (node / "memory.memsw.limit_in_bytes").write_text(str(7 * GIB))
    point_proc(monkeypatch, tmp_path, "9:memory:/scope\n")
    assert CgroupMemory.enforced_limit() == 7 * GIB


def test_root_path_membership_walks_from_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bare `0::/` membership starts the walk at the cgroup root itself."""
    point_cgroup_root(monkeypatch, tmp_path)
    (tmp_path / "memory.max").write_text(str(16 * GIB))
    point_proc(monkeypatch, tmp_path, "0::/\n")
    assert CgroupMemory.enforced_limit() == 16 * GIB


def test_host_exposes_cgroup_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    """`Host.cgroup_memory` defers to the model probe."""
    sentinel = CgroupMemory(limit_bytes=5 * GIB, capped=True)
    monkeypatch.setattr(CgroupMemory, "probe", classmethod(lambda cls: sentinel))
    assert Host().cgroup_memory is sentinel
