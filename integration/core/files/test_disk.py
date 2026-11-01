"""Every large writer keeps the same floor of free disk, wherever the disk is mounted."""

import shutil
from collections.abc import Callable
from pathlib import Path

import pytest
from hypothesis import example, given
from hypothesis import strategies as st

from mainboard import MissionError
from mainboard.core import Disk
from mainboard.core.constants.units import BYTES_PER_GIBIBYTE, DISK_FLOOR

from ...fakes import disks

# A path no machine holds: the disk measured is the one its nearest existing folder sits on.
_NOWHERE = Path("/nowhere/at/all")

_BYTES = st.integers(min_value=0, max_value=4 * DISK_FLOOR)


def _refused(write: Callable[[], None], free: int) -> bool:
    """Whether `write` refuses on a disk with `free` bytes free."""
    with pytest.MonkeyPatch.context() as patched:
        patched.setattr(shutil, "disk_usage", disks(free))
        try:
            write()
        except MissionError:
            return True
    return False


@given(free=_BYTES, need=_BYTES)
@example(free=DISK_FLOOR, need=0)
@example(free=DISK_FLOOR + 1, need=2)
def test_a_writer_is_refused_exactly_when_its_bytes_would_cross_the_floor(
    free: int, need: int
) -> None:
    refused = _refused(lambda: Disk(path=_NOWHERE).require(need, doing="writing"), free)

    assert refused == (free - need < DISK_FLOOR)


@given(free=st.integers(min_value=DISK_FLOOR, max_value=4 * DISK_FLOOR), shares=st.integers(1, 9))
@example(free=33 * BYTES_PER_GIBIBYTE, shares=3)
def test_an_allowance_is_one_share_of_the_space_above_the_floor(free: int, shares: int) -> None:
    with pytest.MonkeyPatch.context() as patched:
        patched.setattr(shutil, "disk_usage", disks(free))
        allowance = Disk(path=_NOWHERE).allowance(shares, doing="collecting")

    assert allowance == (free - DISK_FLOOR) // shares


def test_a_refusal_names_the_writer_the_disk_its_free_space_the_floor_and_what_to_free(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(shutil, "disk_usage", disks(10 * BYTES_PER_GIBIBYTE))

    with pytest.raises(MissionError) as refused:
        Disk(path=_NOWHERE).require(5 * BYTES_PER_GIBIBYTE, doing="copying the tree")

    assert str(refused.value).startswith(
        "copying the tree would leave /nowhere/at/all below the 30 GiB of free disk every large "
        "writer keeps: it has 10.0 GiB free and this writes up to 5.0 GiB. Free 25.0 GiB"
    )


def test_a_path_not_made_yet_is_measured_at_its_nearest_existing_folder(tmp_path: Path) -> None:
    disk = Disk(path=tmp_path / "not" / "made")

    assert (disk.existing, disk.device) == (tmp_path, tmp_path.stat().st_dev)
