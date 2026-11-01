"""The identity vocabulary: one spelling per key, as an attribute and as a Git trailer."""

import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from mainboard.core.events import Identity, Key, Trailer

_SUBJECT = "save 2026-10-28T12:00:00Z"
_COMMIT = f"{_SUBJECT}\n\nTwo files changed.\n\nCo-Authored-By: Claude <noreply@anthropic.com>\n"
_TRAILERS = (
    Trailer(key=Key.host, value="gold"),
    Trailer(key=Key.session, value="claude:0b6e1f"),
    Trailer(key=Key.session, value="codex:019f"),
)

_VALUE = st.text(st.characters(blacklist_categories=("Cc", "Cs", "Z")), min_size=1, max_size=20)
_PROSE = st.text(st.sampled_from("abc xyz."), min_size=1, max_size=40).map(str.strip).filter(bool)
_MESSAGE = st.lists(_PROSE, min_size=1, max_size=3).map("\n\n".join)
_IDENTITY = st.builds(Identity, **{name: _VALUE | st.just("") for name in Identity.model_fields})


@pytest.mark.parametrize(
    ("key", "attribute", "trailer"),
    [
        pytest.param(Key.host, "host.name", "Mb-Host", id="a-machine"),
        pytest.param(Key.workspace, "mb.workspace", "Mb-Workspace", id="a-checkout"),
        pytest.param(Key.session, "session.id", "Mb-Session", id="a-conversation"),
        pytest.param(
            Key.previous_session,
            "session.previous_id",
            "Mb-Previous-Session",
            id="a-continued-thread",
        ),
        pytest.param(Key.job, "mb.job", "Mb-Job", id="a-dispatched-run"),
        pytest.param(Key.revision, "vcs.ref.head.revision", "Mb-Revision", id="a-git-id"),
        pytest.param(Key.object, "mb.object", "Mb-Object", id="a-content-hash"),
    ],
)
def test_each_key_has_one_attribute_spelling_and_one_trailer_spelling(
    key: Key, attribute: str, trailer: str
) -> None:
    assert (str(key), key.trailer, Key.parsed(trailer.upper())) == (attribute, trailer, key)


@settings(max_examples=200)
@given(identity=_IDENTITY)
@example(identity=Identity(session="claude:0b6e1f", previous_session="codex:019f"))
@example(identity=Identity())
def test_an_identity_survives_both_its_carriers(identity: Identity) -> None:
    through_trailers = Identity.from_trailers(
        Trailer.append(_SUBJECT + "\n\nbody", identity.trailers())
    )
    through_attributes = Identity.from_attributes(
        {str(k): v for k, v in identity.attributes().items()}
    )
    assert (through_trailers, through_attributes) == (identity, identity)


@settings(max_examples=200)
@given(message=_MESSAGE)
@example(message=_SUBJECT)
@example(message=f"{_SUBJECT}\n\nMb-Host: not-a-trailer-in-the-subject-block")
def test_trailers_appended_to_a_message_are_read_back_after_what_it_carried(message: str) -> None:
    written = Trailer.append(message, _TRAILERS)
    assert Trailer.read(written)[-len(_TRAILERS) :] == _TRAILERS
    assert written.startswith(message)


def test_a_new_trailer_joins_the_block_a_real_commit_already_has() -> None:
    written = Trailer.append(_COMMIT, _TRAILERS[:1])
    assert written == _COMMIT + "Mb-Host: gold\n"
    assert Trailer.read(written) == _TRAILERS[:1]


@pytest.mark.parametrize(
    "message",
    [
        pytest.param(f"{_SUBJECT}\n\nSigned-off-by: A <a@b.c>\n", id="trailers-that-are-not-mb-s"),
        pytest.param(f"{_SUBJECT}\n\nSession: claude:0b6e1f\n", id="a-key-without-the-mb-prefix"),
        pytest.param("Mb-Host: gold\n", id="a-subject-line-that-spells-a-trailer"),
        pytest.param(f"{_SUBJECT}\n\nMb-Host: gold\n\nand then prose", id="a-trailer-above-prose"),
        pytest.param("", id="an-empty-message"),
    ],
)
def test_text_that_is_not_a_trailer_block_yields_no_mb_trailers(message: str) -> None:
    assert Trailer.read(message) == ()


@pytest.mark.parametrize("value", ["", " padded ", "two\nlines", "a\rb"])
def test_a_trailer_value_is_one_trimmed_line(value: str) -> None:
    with pytest.raises(ValueError, match="should match pattern"):
        Trailer(key=Key.host, value=value)


@pytest.mark.parametrize("token", ["Signed-off-by", "Session", "Mb-Sessions"])
def test_a_token_that_does_not_spell_an_mb_key_is_not_one(token: str) -> None:
    assert Key.parsed(token) is None
