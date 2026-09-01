import os
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from mainboard.batch import Transfer, TransferSet
from mainboard.dispatch import HostSetup

from .support import spec

if TYPE_CHECKING:
    from mainboard import Board

# A watermark and the two sides of it, as epoch seconds a test can stamp a file with.
_MIRRORED = "2026-08-19T00:00:00+00:00"
_BEFORE = 1787011200.0  # a day under the watermark
_AFTER = 1787184000.0  # a day over it


def written(root: Path, name: str, *, size: int = 4096, at: float = _AFTER) -> Path:
    """One compressible file at `name`, stamped as last changed at `at`."""
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((b"mainboard " * (size // 10 + 1))[:size])
    os.utime(path, (at, at))
    return path


def mirrored(board: Board, alias: str = "miyabi-g", *, at: str = _MIRRORED) -> None:
    """Record that `alias` was onboarded, and therefore mirrored, at `at`."""
    board.dispatcher.cache.save_host(HostSetup(host=alias, root="/work/p"))
    board.dispatcher.cache.connection.execute(
        "UPDATE hosts SET facts = json_set(facts, '$.onboarded_at', ?) WHERE alias = ?",
        (at, alias),
    )


def measured(board: Board, **job: str | int) -> TransferSet:
    """The transfer set for one declared job over `board`."""
    return Transfer(board).set_for(spec({"target": "miyabi-g", "command": "true", **job}).jobs[0])


def test_a_job_on_this_machine_ships_nothing_at_all(lab: Board) -> None:
    """The workspace is already here, so a local job has no wire to cross."""
    written(lab.root, "packages/core/train.py")
    empty = Transfer(lab).set_for(spec({"target": "local", "command": "true"}).jobs[0])
    assert (empty.files, empty.raw_bytes, empty.wire_bytes, empty.since) == (0, 0, 0, "")


def test_a_target_with_no_recorded_mirror_ships_everything_in_scope(lab: Board) -> None:
    """Nothing to subtract means everything the include scope names is in flight, and says so."""
    written(lab.root, "packages/core/train.py", at=_BEFORE)
    written(lab.root, "packages/core/data.bin", at=_AFTER)
    set_for = measured(lab)
    assert (set_for.files, set_for.since) == (2, "")
    assert set_for.raw_bytes == 8192
    assert 0 < set_for.wire_bytes < set_for.raw_bytes
    assert set_for.paths == ("packages",)


def test_only_what_changed_since_the_mirror_is_counted_as_in_flight(lab: Board) -> None:
    """The host already carries the rest, so shipping it again is not what this job costs."""
    written(lab.root, "packages/core/train.py", at=_BEFORE)
    written(lab.root, "packages/core/fresh.py", at=_AFTER)
    mirrored(lab)
    set_for = measured(lab)
    assert (set_for.files, set_for.since) == (1, _MIRRORED)
    assert set_for.raw_bytes == 4096


def test_the_data_a_job_names_ships_whether_or_not_the_mirror_would_carry_it(lab: Board) -> None:
    """A dataset outside the include scope is exactly what a job has to say it needs."""
    written(lab.root, "packages/core/train.py", at=_BEFORE)
    written(lab.root, "corpus/shard.npz", at=_BEFORE)
    mirrored(lab)
    set_for = measured(lab, data=["corpus"])
    assert (set_for.files, set_for.raw_bytes) == (1, 4096)
    assert set_for.paths == ("packages", "corpus")


def test_a_file_the_job_names_directly_ships_without_a_directory_around_it(lab: Board) -> None:
    written(lab.root, "corpus/shard.npz", at=_BEFORE)
    mirrored(lab)
    assert measured(lab, data=["corpus/shard.npz", "corpus/missing.npz"]).files == 1


def test_nothing_the_mirror_refuses_is_ever_counted(lab: Board) -> None:
    """The denylist, the host's own excludes and every `.gitignore`, as rsync itself sees them."""
    (lab.root / ".gitignore").write_text("*.log\n")
    written(lab.root, "packages/core/train.py")
    written(lab.root, "packages/core/run.log")
    written(lab.root, "packages/core/__pycache__/train.pyc")
    written(lab.root, "packages/.pixi/envs/default/bin/python")
    written(lab.root, "packages/data/raw/dump.bin")
    written(lab.root, "packages/rust/target/release/lib.rlib")
    (lab.root / "packages/rust/.gitignore").write_text("target/\n")
    # The nested ignore file itself ships, since the mirror hands it to the receiver to prune with.
    assert measured(lab).files == 2


def test_a_symlink_is_never_followed_into_a_second_copy_of_the_tree(lab: Board) -> None:
    """A link back into the workspace would otherwise be measured again under its own name."""
    written(lab.root, "packages/core/train.py")
    try:
        (lab.root / "packages/loop").symlink_to(
            lab.root / "packages/core", target_is_directory=True
        )
    except OSError as fault:
        pytest.skip(f"this Windows account cannot create symlinks: {fault}")
    (lab.root / "packages/dangling").symlink_to(lab.root / "packages/gone")
    assert measured(lab).files == 1


def test_the_ignore_rules_of_a_directory_are_read_once_however_deep_the_walk(lab: Board) -> None:
    written(lab.root, "packages/core/a.py")
    written(lab.root, "packages/core/b.py")
    transfer = Transfer(lab)
    transfer.set_for(spec({"target": "miyabi-g", "command": "true"}).jobs[0])
    assert Path("packages/core") in transfer.nested
    assert transfer.rules(Path("packages/core")) is transfer.rules(Path("packages/core"))


def test_a_measurement_streams_a_file_larger_than_one_read(lab: Board) -> None:
    """A dataset never has to fit in memory to be counted, so the reader is a loop, not a slurp."""
    written(lab.root, "packages/core/big.bin", size=3 << 20)
    assert measured(lab).raw_bytes == 3 << 20


@pytest.mark.parametrize("alias", ["miyabi-g", "nowhere"])
def test_a_host_the_registry_never_recorded_has_no_watermark_to_subtract(
    lab: Board, alias: str
) -> None:
    assert Transfer(lab).watermark(alias) == ""
