"""The spool of keeps: what waits is what was appended and not yet delivered, oldest first."""

import stat
import string
from collections.abc import Sequence
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from hypothesis import given
from hypothesis import strategies as st

from mainboard import MissionError
from mainboard.core import CorruptFile
from mainboard.memory.keeps import Keep
from mainboard.memory.outbox import Outbox

_WORDS = st.text(alphabet=string.ascii_letters, min_size=1, max_size=12)
_BYTES = b"\x89PNG\x00\xff\xfe\x01"


@given(words=st.lists(_WORDS, unique=True, max_size=6), delivered=st.integers(0, 6))
def test_what_waits_is_what_was_appended_and_not_delivered_oldest_first(
    words: Sequence[str], delivered: int
) -> None:
    keeps = [Keep.of(word) for word in words]
    with TemporaryDirectory() as root:
        outbox = Outbox(Path(root))
        for keep in keeps:
            outbox.append(keep)
        for keep in keeps[:delivered]:
            outbox.settled(keep, "document")
        assert [waiting.keep for waiting in outbox.waiting()] == keeps[delivered:]


def test_a_keep_appended_twice_or_after_its_delivery_is_spooled_once(workspace: Path) -> None:
    outbox, keep = Outbox(workspace), Keep.of("a note")
    outbox.append(keep)
    outbox.append(keep)
    waiting = len(outbox.waiting())
    outbox.settled(keep, "document-1")
    outbox.append(keep)
    receipt = outbox.receipt(keep.key)
    assert (waiting, outbox.waiting(), receipt and receipt.document) == (1, [], "document-1")


def test_a_binary_file_waits_as_a_copy_and_leaves_with_its_receipt(
    workspace: Path, tmp_path: Path
) -> None:
    picture = tmp_path / "shot.png"
    picture.write_bytes(_BYTES)
    outbox = Outbox(workspace)
    outbox.append(Keep.of(str(picture)))
    picture.unlink()
    (waiting,) = outbox.waiting()
    held = waiting.keep.source
    assert held is not None and held.read_bytes() == _BYTES
    outbox.settled(waiting.keep, "document-1")
    assert (held.exists(), outbox.waiting()) == (False, [])


def test_a_file_that_changed_since_it_was_digested_is_refused_and_leaves_nothing(
    workspace: Path, tmp_path: Path
) -> None:
    picture = tmp_path / "shot.png"
    picture.write_bytes(_BYTES)
    keep = Keep.of(str(picture))
    picture.write_bytes(_BYTES + b"\x02")
    outbox = Outbox(workspace)
    with pytest.raises(MissionError, match="changed while"):
        outbox.append(keep)
    assert (outbox.waiting(), list(outbox.spool.glob("*.blob"))) == ([], [])


def test_a_spool_line_of_another_keep_is_set_aside_and_never_delivered(workspace: Path) -> None:
    outbox, first, second = Outbox(workspace), Keep.of("one"), Keep.of("two")
    outbox.append(first)
    outbox.spool.joinpath(f"{second.key}.jsonl").write_bytes(
        outbox.spool.joinpath(f"{first.key}.jsonl").read_bytes()
    )
    assert (len(outbox.waiting()), len(list(outbox.spool.glob("*.refused")))) == (1, 1)


def test_a_folder_named_as_a_spool_line_is_set_aside_and_the_keeps_beside_it_wait(
    workspace: Path,
) -> None:
    outbox, keep = Outbox(workspace), Keep.of("one")
    outbox.append(keep)
    outbox.spool.joinpath(f"{Keep.of('two').key}.jsonl").mkdir()
    waiting = [waiting.keep for waiting in outbox.waiting()]
    assert (waiting, len(list(outbox.spool.glob("*.refused")))) == ([keep], 1)


def test_a_receipt_that_does_not_read_as_one_is_refused_as_a_corrupt_file(
    workspace: Path,
) -> None:
    outbox, keep = Outbox(workspace), Keep.of("one")
    outbox.settled(keep, "document-1")
    outbox.kept.joinpath(f"{keep.key}.json").write_text("{not a receipt", encoding="utf-8")
    with pytest.raises(CorruptFile, match="move it aside"):
        outbox.append(keep)


def test_a_second_pass_delivering_at_once_is_told_to_stand_aside(workspace: Path) -> None:
    outbox = Outbox(workspace)
    with outbox.delivering() as first, outbox.delivering() as second:
        assert (first, second) == (True, False)
    with outbox.delivering() as again:
        assert again is True


def test_the_pass_that_delivers_keeps_its_lock_file_private(workspace: Path) -> None:
    outbox = Outbox(workspace)
    with outbox.delivering():
        pass
    assert stat.S_IMODE(outbox.lock.stat().st_mode) == 0o600
