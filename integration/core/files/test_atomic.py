"""`write_atomic` and `replacing` put a file in place whole and leave nothing stray beside it."""

import os
import stat
from collections.abc import Iterator
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from hypothesis import example, given
from hypothesis import strategies as st

from mainboard.core import replacing, write_atomic


@pytest.fixture
def strict_umask() -> Iterator[None]:
    previous = os.umask(0o077)
    yield
    os.umask(previous)


@given(
    old=st.binary(max_size=32),
    text=st.text(max_size=32),
    mode=st.none() | st.sampled_from([0o600, 0o640, 0o644]),
    durable=st.booleans(),
)
@example(old=b"old", text="é\r\n", mode=0o600, durable=True)
def test_a_write_replaces_the_file_whole_and_leaves_nothing_beside_it(
    old: bytes, text: str, mode: int | None, durable: bool
) -> None:
    with TemporaryDirectory() as root:
        path = Path(root) / "held.txt"
        path.write_bytes(old)

        write_atomic(path, text, mode=mode, durable=durable)

        assert (path.read_bytes(), os.listdir(root)) == (text.encode("utf-8"), ["held.txt"])


@pytest.mark.usefixtures("strict_umask")
@pytest.mark.parametrize(
    ("mode", "held"),
    [
        pytest.param(0o644, 0o644, id="an-exact-mode-whatever-the-umask"),
        pytest.param(None, 0o600, id="no-mode-leaves-it-to-the-umask"),
    ],
)
def test_a_given_mode_holds_whatever_the_umask_and_none_follows_it(
    tmp_path: Path, mode: int | None, held: int
) -> None:
    path = tmp_path / "held.bin"

    write_atomic(path, b"\xff", mode=mode)

    assert stat.S_IMODE(path.stat().st_mode) == held


@pytest.mark.parametrize(
    ("durable", "synced"),
    [
        pytest.param(True, 1, id="a-durable-write-syncs"),
        pytest.param(False, 0, id="by-default-not"),
    ],
)
def test_only_a_durable_write_syncs_the_bytes_before_the_rename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, durable: bool, synced: int
) -> None:
    calls: list[int] = []
    monkeypatch.setattr(os, "fsync", calls.append)

    write_atomic(tmp_path / "synced.txt", "x", durable=durable)

    assert len(calls) == synced


def test_a_rename_that_fails_leaves_the_old_target_and_no_stray_file(tmp_path: Path) -> None:
    target = tmp_path / "held"
    target.mkdir()
    (target / "inner").write_text("kept", encoding="utf-8")

    with pytest.raises(OSError):
        write_atomic(target, "new")

    assert (os.listdir(tmp_path), os.listdir(target)) == (["held"], ["inner"])


def test_a_block_that_raises_leaves_the_old_file_and_removes_the_temporary(tmp_path: Path) -> None:
    path = tmp_path / "stale.txt"
    path.write_text("old", encoding="utf-8")

    with pytest.raises(RuntimeError, match="boom"), replacing(path) as temporary:
        temporary.write_text("new", encoding="utf-8")
        raise RuntimeError("boom")

    assert (path.read_text(encoding="utf-8"), os.listdir(tmp_path)) == ("old", ["stale.txt"])
