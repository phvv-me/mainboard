from pathlib import Path
from typing import Protocol

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from mainboard.probe import CgroupMemory
from mainboard.probe.facts import cgroup_memory as cgroup_mod

_GIB = 1024**3
_HOST_RAM = 64 * _GIB
_UNLIMITED_V1 = "9223372036854771712"  # the kernel's page-aligned near-2**63 "no limit" sentinel

# A cap file holds either a real ceiling or one of the two spellings of unlimited, and the walk
# has to read past both of those to the tightest real number anywhere above the process.
CAP_VALUES = st.lists(
    st.one_of(st.integers(1, 2**50), st.just("max"), st.just(_UNLIMITED_V1)),
    min_size=1,
    max_size=4,
)

# One cgroup layout per case, the membership line `/proc/self/cgroup` holds paired with the cap
# files the tree carries under it, keyed by a node path relative to the cgroup root.
_V2_ANCESTOR_CAP = (
    "0::/job.123/task\n",
    {
        "job.123": {"memory.max": str(100 * _GIB)},  # the jobid scope carries the real cap
        "job.123/task": {"memory.max": "max"},  # the process's own leaf is uncapped
    },
    100 * _GIB,
)
_V1_MEMSW_ANCESTOR_CAP = (
    "5:memory:/pbspro/42.miyabi/0\n4:cpu,cpuacct:/pbspro/42.miyabi/0\n",
    {
        "memory/pbspro/42.miyabi": {
            "memory.limit_in_bytes": str(120 * _GIB),
            "memory.memsw.limit_in_bytes": str(110 * _GIB),  # the tighter, real ceiling
        },
        "memory/pbspro/42.miyabi/0": {
            "memory.limit_in_bytes": _UNLIMITED_V1,
            "memory.memsw.limit_in_bytes": _UNLIMITED_V1,
        },
    },
    110 * _GIB,
)
_V1_CORRUPT_CAP_FILE = (
    "9:memory:/scope\n",
    {
        "memory/scope": {
            "memory.limit_in_bytes": "garbage",
            "memory.memsw.limit_in_bytes": str(7 * _GIB),
        }
    },
    7 * _GIB,
)
_V2_ROOT_MEMBERSHIP = ("0::/\n", {"": {"memory.max": str(16 * _GIB)}}, 16 * _GIB)
_V2_UNCAPPED = ("0::/user.slice\n", {"user.slice": {"memory.max": "max"}}, None)
_V1_NO_MEMORY_CONTROLLER = ("3:cpu,cpuacct:/some/path\n", {}, None)
_NO_MEMBERSHIP_FILE = (None, {}, None)


class LayOutCgroup(Protocol):
    """Write one cgroup layout, its membership file included, into the fake tree."""

    def __call__(self, membership: str | None, nodes: dict[str, dict[str, str]]) -> None: ...


@pytest.fixture
def lay_out_cgroup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> LayOutCgroup:
    """Point the cgroup tree, the membership file, and the host RAM fallback at a tmp tree.

    The membership file starts out absent, which is what a non-Linux host reads, and a layout
    that passes one writes it into place.
    """
    monkeypatch.setattr(cgroup_mod, "_CGROUP_ROOT", tmp_path)
    monkeypatch.setattr(cgroup_mod, "_CGROUP_PROC", tmp_path / "absent-self-cgroup")
    monkeypatch.setattr(
        cgroup_mod.psutil, "virtual_memory", lambda: type("VM", (), {"total": _HOST_RAM})()
    )

    def lay_out(membership: str | None, nodes: dict[str, dict[str, str]]) -> None:
        if membership is not None:
            proc = tmp_path / "self.cgroup"
            proc.write_text(membership)
            monkeypatch.setattr(cgroup_mod, "_CGROUP_PROC", proc)
        for relative, files in nodes.items():
            node = tmp_path / relative
            node.mkdir(parents=True, exist_ok=True)
            for name, value in files.items():
                (node / name).write_text(value)

    return lay_out


@pytest.mark.parametrize(
    ("membership", "nodes", "expected"),
    [
        pytest.param(*_V2_ANCESTOR_CAP, id="v2-ancestor-cap"),
        pytest.param(*_V1_MEMSW_ANCESTOR_CAP, id="v1-memsw-ancestor-cap"),
        pytest.param(*_V1_CORRUPT_CAP_FILE, id="v1-corrupt-cap-file"),
        pytest.param(*_V2_ROOT_MEMBERSHIP, id="v2-root-membership"),
        pytest.param(*_V2_UNCAPPED, id="v2-uncapped"),
        pytest.param(*_V1_NO_MEMORY_CONTROLLER, id="v1-no-memory-controller"),
        pytest.param(*_NO_MEMBERSHIP_FILE, id="no-membership-file"),
    ],
)
def test_the_enforced_cap_is_read_from_whichever_cgroup_version_the_host_runs(
    membership: str | None,
    nodes: dict[str, dict[str, str]],
    expected: int | None,
    lay_out_cgroup: LayOutCgroup,
) -> None:
    """A scheduler writes the limit on an ancestor of the process's own leaf, so both versions
    are walked up to the root. v2 reads the unified `0::` line and `memory.max`, v1 reads the
    memory-controller line and both `limit_in_bytes` files, and the memsw one is the ceiling
    Miyabi's GH200 PBS actually enforces. A corrupt file, an unlimited sentinel, a membership
    line for another controller, and an unreadable membership file all contribute nothing."""
    lay_out_cgroup(membership, nodes)

    assert CgroupMemory.enforced_limit() == expected
    cgroup = CgroupMemory.probe()
    assert cgroup.capped is (expected is not None)
    assert cgroup.limit_bytes == (expected if expected is not None else _HOST_RAM)
    assert cgroup.limit_gb == cgroup.limit_bytes / _GIB


# Each example builds a whole cgroup chain on disk, so the budget is trimmed to keep the
# suite's inner loop fast. Every branch is pinned by the layouts above rather than found here.
@settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(caps=CAP_VALUES)
def test_the_tightest_real_ceiling_anywhere_on_the_walk_is_the_one_reported(
    caps: list[int | str], lay_out_cgroup: LayOutCgroup
) -> None:
    """The process can sit many levels below the node the limit was written on and any level may
    be unlimited, so the answer is the smallest finite value found on the whole chain, while an
    all-unlimited chain reports the host's own RAM rather than a sentinel near 2**63."""
    # Each depth gets its own subtree, so one example never reads a level a longer one wrote.
    levels = [f"depth{len(caps)}"] + [f"level{index}" for index in range(len(caps))]
    nodes = {
        "/".join(levels[: index + 2]): {"memory.max": str(cap)} for index, cap in enumerate(caps)
    }
    lay_out_cgroup(f"0::/{'/'.join(levels)}\n", nodes)

    finite = [cap for cap in caps if isinstance(cap, int)]
    assert CgroupMemory.enforced_limit() == (min(finite) if finite else None)
    assert CgroupMemory.probe().limit_bytes == (min(finite) if finite else _HOST_RAM)
