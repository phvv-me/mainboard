import codecs
from typing import TYPE_CHECKING

import pytest
from hypothesis import given
from hypothesis import strategies as st

from mainboard.lint.inventory import Attributes
from mainboard.lint.text import decoded, normalized, problems, repair

if TYPE_CHECKING:
    from pathlib import Path

# Text with every newline style and trailing blank a real file arrives with. NUL is left out so a
# UTF-8 encoding of it is never taken for binary, and a leading U+FEFF so it is never a BOM.
_MESSY = st.lists(
    st.one_of(
        st.sampled_from(["\r\n", "\r", "\n", " ", "\t", "  \n"]),
        st.characters(codec="utf-8", exclude_categories=("Cs",), exclude_characters="\0\ufeff"),
    ),
    max_size=60,
).map("".join)
_NEWLINES = st.sampled_from(["\n", "\r\n"])


@given(text=_MESSY, newline=_NEWLINES)
def test_normalized_text_is_a_fixed_point_with_one_newline_style_and_one_final_newline(
    text: str, newline: str
) -> None:
    fixed = normalized(text, newline)

    assert normalized(fixed, newline) == fixed
    lines = fixed.split(newline)
    assert "\r" not in "".join(lines)
    assert all(line == line.rstrip(" \t") for line in lines)
    assert fixed == "" or (fixed.endswith(newline) and not fixed.endswith(newline * 2))


@given(text=_MESSY)
def test_normalizing_moves_only_whitespace(text: str) -> None:
    assert normalized(text).split() == text.split()


@given(text=_MESSY)
def test_every_unicode_encoding_a_windows_editor_writes_decodes_to_the_same_text(
    text: str,
) -> None:
    assert decoded(text.encode("utf-16")) == text
    assert decoded(codecs.BOM_UTF16_BE + text.encode("utf-16-be")) == text
    assert decoded(codecs.BOM_UTF8 + text.encode()) == text
    assert decoded(text.encode()) == text


def test_binary_is_left_alone_and_a_legacy_code_page_is_refused() -> None:
    assert decoded(b"PK\x03\x04\0\0data") is None
    with pytest.raises(UnicodeDecodeError):
        decoded("café".encode("latin-1"))


@pytest.mark.parametrize(
    ("name", "text", "expected"),
    [
        ("merge.py", "<<<<<<< ours\na\n=======\nb\n>>>>>>> theirs\n", ["merge conflict"]),
        ("heading.rst", "Title\n=======\n", []),
        ("key.txt", "-----BEGIN OPENSSH PRIVATE KEY-----\n", ["private key"]),
        ("putty.ppk", "PuTTY-User-Key-File-3: ssh-ed25519\n", ["private key"]),
        ("bad.toml", "k = [\n", ["does not parse"]),
        ("good.toml", "k = [1]\n", []),
        ("bad.yaml", "a: [\n", ["does not parse"]),
        ("tagged.yml", "job: !reference [a, b]\n---\nsecond: 1\n", []),
        ("plain.md", "a: [\n", []),
    ],
    ids=[
        "both conflict halves",
        "a lone underline is not a conflict",
        "a private key",
        "a PuTTY key",
        "broken TOML",
        "valid TOML",
        "broken YAML",
        "custom tags and several documents parse",
        "a document is never parsed as data",
    ],
)
def test_problems_name_what_only_a_person_can_fix(
    name: str, text: str, expected: list[str]
) -> None:
    found = problems(name, text)

    assert len(found) == len(expected)
    assert all(fragment in line for fragment, line in zip(expected, found, strict=True))


@pytest.mark.parametrize(
    ("content", "attributes", "stored", "left"),
    [
        (b"a  \r\nb\r\n\r\n", Attributes(), b"a\nb\n", []),
        (b"a\nb", Attributes(newline="\r\n"), b"a\r\nb\r\n", []),
        ("x = 1\n".encode("utf-16"), Attributes(), b"x = 1\n", []),
        (b"a  \n", Attributes(binary=True), b"a  \n", []),
        (b"\0\0 \n", Attributes(), b"\0\0 \n", []),
        ("café \n".encode("latin-1"), Attributes(), "café \n".encode("latin-1"), ["UTF-8"]),
        (b"k = [\n", Attributes(), b"k = [\n", ["does not parse"]),
    ],
    ids=[
        "CRLF and trailing blanks",
        "an eol=crlf file keeps CRLF",
        "UTF-16 becomes UTF-8",
        "a -text file is untouched",
        "binary content is untouched",
        "a legacy code page is refused, not guessed",
        "a syntax error survives the repair",
    ],
)
def test_repair_rewrites_only_what_it_can_decide_and_names_the_rest(
    tmp_path: Path, content: bytes, attributes: Attributes, stored: bytes, left: list[str]
) -> None:
    path = tmp_path / ("data.toml" if b"k = " in content else "notes.txt")
    path.write_bytes(content)

    found = repair(path, attributes)

    assert path.read_bytes() == stored
    assert [any(fragment in line for line in found) for fragment in left] == [True] * len(left)
    assert len(found) == len(left)


def test_repair_never_writes_through_a_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target.txt"
    target.write_bytes(b"a  \n")
    link = tmp_path / "link.txt"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("this account may not create symlinks, as on Windows without developer mode")

    assert repair(link, Attributes()) == []
    assert target.read_bytes() == b"a  \n"
