"""A keep: what an item is, and the key that makes a replay one keep."""

import os
import string
from datetime import UTC, datetime
from pathlib import Path

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from mainboard import MissionError
from mainboard.memory.enums import Kind
from mainboard.memory.keeps import Keep

_WORDS = st.text(alphabet=string.ascii_letters + " ", min_size=1, max_size=40)
_SCOPES = st.text(alphabet=string.ascii_lowercase, max_size=6)
_NOON = datetime(2026, 10, 29, 12, 0, tzinfo=UTC)


@given(text=_WORDS, first=_WORDS, second=_WORDS, scope=_SCOPES, other=_SCOPES)
def test_a_key_names_the_content_and_the_scope_whatever_the_note(
    text: str, first: str, second: str, scope: str, other: str
) -> None:
    assume(scope != other)
    kept = Keep.of(text, note=first, scope=scope)
    assert Keep.of(text, note=second, scope=scope).key == kept.key
    assert Keep.of(text, note=first, scope=other).key != kept.key


@given(text=_WORDS, scope=_SCOPES)
def test_a_keep_survives_its_spool_line_and_names_the_line_by_its_key(
    text: str, scope: str
) -> None:
    keep = Keep.of(text, scope=scope)
    event = keep.event(_NOON)
    assert (Keep.from_event(event), event.id) == (keep, keep.key)


def test_a_text_file_is_kept_as_the_text_it_holds(tmp_path: Path) -> None:
    note = tmp_path / "note.md"
    note.write_text("# a note\n", encoding="utf-8")
    assert Keep.of(str(note)).key == Keep.of("# a note\n").key


@pytest.mark.parametrize(
    ("item", "kind"),
    [
        pytest.param("a thought about gold", Kind.TEXT, id="plain-words"),
        pytest.param("TCP/IP", Kind.TEXT, id="a-slash-without-a-suffix"),
        pytest.param("https://example.com/a", Kind.URI, id="a-uri"),
    ],
)
def test_an_item_that_names_no_file_is_text_or_a_uri(item: str, kind: Kind) -> None:
    assert Keep.of(item).kind is kind


def test_a_text_file_is_shown_by_its_name_never_by_what_it_holds(tmp_path: Path) -> None:
    note = tmp_path / "plan.md"
    note.write_text("the first line of the plan\n", encoding="utf-8")
    assert Keep.of(str(note)).label == "plan.md"


@pytest.mark.parametrize(
    ("place", "deny"),
    [
        pytest.param("work/.env", "./.env", id="a-dotenv"),
        pytest.param("work/.env.local", "./.env.*", id="a-dotenv-variant"),
        pytest.param("work/.mb/serve/token", "./.mb/serve/**", id="the-serve-state"),
        pytest.param("home/.ssh/id_rsa", "~/.ssh/**", id="an-ssh-key"),
        pytest.param("home/.config/mainboard/grants/g.json", "grants/**", id="the-grants"),
    ],
)
def test_a_file_that_may_hold_secrets_is_refused_without_a_word_of_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, place: str, deny: str
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    secret = tmp_path / place
    secret.parent.mkdir(parents=True)
    secret.write_text("OPENAI_API_KEY=sk-canary\n", encoding="utf-8")
    beside = tmp_path / "notes.txt"
    beside.symlink_to(secret)
    for item in (str(secret), str(beside)):
        with pytest.raises(MissionError, match="may hold secrets") as refused:
            Keep.of(item)
        assert deny in str(refused.value) and "canary" not in str(refused.value)


def test_a_pipe_is_refused_at_once_without_waiting_on_it(tmp_path: Path) -> None:
    pipe = tmp_path / "pipe.txt"
    os.mkfifo(pipe)
    with pytest.raises(MissionError, match="no regular file"):
        Keep.of(str(pipe))


def test_a_file_that_is_no_text_is_kept_as_its_bytes_by_its_name(tmp_path: Path) -> None:
    picture = tmp_path / "shot.png"
    picture.write_bytes(b"\x89PNG\x00\xff\xfe")
    keep = Keep.of(str(picture), note="a screenshot")
    assert (keep.kind, keep.content, keep.source) == (Kind.FILE, "shot.png", picture.resolve())


@pytest.mark.parametrize(
    ("item", "said"),
    [
        pytest.param("notes/gone.md", "is no file", id="a-path-to-nothing"),
        pytest.param("~/gone.md", "is no file", id="a-home-path-to-nothing"),
        pytest.param(".", "is a folder", id="a-folder"),
    ],
)
def test_an_item_that_names_a_place_and_no_file_is_refused(item: str, said: str) -> None:
    with pytest.raises(MissionError, match=said):
        Keep.of(item)
