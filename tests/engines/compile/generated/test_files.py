from typing import TYPE_CHECKING

from mainboard.engines.compile.generated import GeneratedFiles, Writer

if TYPE_CHECKING:
    from pathlib import Path


def test_locked_creates_the_directory_and_shares_one_reentrant_lock(tmp_path: Path) -> None:
    """`Provisioner.provision` holds it around a `Compiler.stale` taking it again."""
    directory = tmp_path / ".mainboard"
    with GeneratedFiles(directory=directory).locked() as outer:
        assert directory.is_dir()
        assert isinstance(outer, Writer)
        with GeneratedFiles(directory=directory).locked() as inner:
            assert inner.lock is outer.lock
            assert inner.lock.is_locked
        assert outer.lock.is_locked
    assert not outer.lock.is_locked
