from typing import TYPE_CHECKING

from mainboard.engines.compile.generated import GeneratedFiles, Writer

if TYPE_CHECKING:
    from pathlib import Path


def test_locked_creates_the_directory_and_hands_out_a_writer(tmp_path: Path) -> None:
    directory = tmp_path / ".mainboard"
    with GeneratedFiles(directory=directory).locked() as writer:
        assert directory.is_dir()
        assert isinstance(writer, Writer)
        assert writer.lock.is_locked


def test_locked_is_reentrant_across_separate_instances_on_the_same_directory(
    tmp_path: Path,
) -> None:
    """The shared lock registry lets a nested `locked()` on the same dir avoid deadlocking.

    `Provisioner.provision` opens the lock once around the whole install, and `Compiler.stale`
    reads state through the same directory while that outer lock is still held, so two
    `GeneratedFiles` instances on the same path must share one reentrant lock.
    """
    directory = tmp_path / ".mainboard"
    with GeneratedFiles(directory=directory).locked() as outer:
        with GeneratedFiles(directory=directory).locked() as inner:
            assert inner.lock is outer.lock
            assert inner.lock.is_locked
        assert outer.lock.is_locked
    assert not outer.lock.is_locked
