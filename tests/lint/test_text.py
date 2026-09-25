import codecs
from typing import TYPE_CHECKING

import pytest
from hypothesis import given
from hypothesis import strategies as st

from mainboard.lint.inventory import Attributes
from mainboard.lint.text import Examination, decoded, examined, normalized, problems, untidiness

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


@given(text=_MESSY, newline=_NEWLINES, encoding=st.sampled_from(["utf-8", "utf-8-sig", "utf-16"]))
def test_the_untidiness_named_is_empty_exactly_when_normalizing_changes_no_byte(
    text: str, newline: str, encoding: str
) -> None:
    data = text.encode(encoding)

    assert (not untidiness(data, text, newline)) == (normalized(text, newline).encode() == data)


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
    ("content", "attributes", "untidy", "left"),
    [
        (b"a  \r\nb\r\n\r\n", Attributes(), ["line endings", "trailing", "at the end"], []),
        (b"a\nb", Attributes(newline="\r\n"), ["line endings", "at the end"], []),
        ("x = 1\n".encode("utf-16"), Attributes(), ["UTF-8"], []),
        (b"a\n", Attributes(), [], []),
        (b"a  \n", Attributes(binary=True), [], []),
        (b"\0\0 \n", Attributes(), [], []),
        ("caf\u00e9 \n".encode("latin-1"), Attributes(), [], ["UTF-8"]),
        (b"k = [  \n", Attributes(), ["trailing"], ["does not parse"]),
    ],
    ids=[
        "CRLF and trailing blanks",
        "an eol=crlf file keeps CRLF",
        "UTF-16 becomes UTF-8",
        "tidy text",
        "a -text file is untouched",
        "binary content is untouched",
        "a legacy code page is refused, not guessed",
        "a syntax error survives the repair",
    ],
)
def test_an_examination_names_the_repair_it_would_make_and_what_is_left_writing_nothing(
    tmp_path: Path, content: bytes, attributes: Attributes, untidy: list[str], left: list[str]
) -> None:
    path = tmp_path / ("data.toml" if b"k = " in content else "notes.txt")
    path.write_bytes(content)

    found = examined(path, attributes)

    assert path.read_bytes() == content
    assert len(found.untidy) == len(untidy)
    assert all(part in change for part, change in zip(untidy, found.untidy, strict=True))
    assert (found.repaired is None) == (not untidy)
    assert len(found.problems) == len(left)
    assert all(part in problem for part, problem in zip(left, found.problems, strict=True))


def test_an_examination_never_reads_through_a_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target.txt"
    target.write_bytes(b"a  \n")
    link = tmp_path / "link.txt"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("this account may not create symlinks, as on Windows without developer mode")

    assert examined(link, Attributes()) == Examination()
