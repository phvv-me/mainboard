import pytest
from hypothesis import given
from hypothesis import strategies as st

from mainboard import MissionError
from mainboard.deps import ManifestText

from ..strategies import SPECS, WORDS

_DEV_PYTHON = ("dev", "python", "deps")
_PYTHON = ("python", "deps")

# Every dependency table the fixture manifest already carries, so a generated entry joins
# neighbours rather than creating a heading the round trip would then have to leave behind.
_TABLES = st.sampled_from(
    [
        ("deps",),
        _PYTHON,
        ("dev", "deps"),
        _DEV_PYTHON,
        ("nodejs", "deps"),
        ("nodejs", "dev"),
        ("envs", "serving", "python", "deps"),
    ]
)

# What the fixture already declares, kept out of the generated names so an add is always an
# arrival and the drop that follows it is always a departure.
_DECLARED = frozenset(
    {
        "python",
        "pueue",
        "torch",
        "protobuf",
        "pytest",
        "sglang",
        "vllm",
        "vite",
        "flashinfer",
    }
)
_NEW = WORDS.filter(lambda word: word not in _DECLARED)


@given(path=_TABLES, name=_NEW, spec=SPECS)
def test_adding_a_requirement_and_dropping_it_leaves_the_file_byte_identical(
    text: str, path: tuple[str, ...], name: str, spec: str
) -> None:
    """The round trip is the proof that nothing but the one line was ever touched."""
    edited = ManifestText(text)
    edited.put(path, name, spec=spec)
    edited.put(path, name, spec=spec)
    assert edited.declares(path, name)
    assert edited.constraint(path, name) == spec
    restored = ManifestText(edited.text())
    restored.drop(path, name)
    assert restored.text() == text


def test_a_new_entry_keeps_the_column_and_the_comment_that_introduces_the_next_table(
    text: str,
) -> None:
    """Alignment is read off the table, and a heading comment stays with the table it announces."""
    edited = ManifestText(text)
    edited.put(_DEV_PYTHON, "tqdm", spec=">=4.70.0, <5")
    written = edited.text().splitlines()
    landed = written.index('tqdm      = ">=4.70.0, <5"')
    assert written[landed - 1].startswith("pytest")
    assert (
        written[landed + 2]
        == "# runtime-keyed toolchains, the table name matches the package in [deps]"
    )
    aligned = ManifestText(text)
    aligned.put(_PYTHON, "tqdm", spec=">=4")
    columns = {
        line.index("=")
        for line in aligned.text().splitlines()
        if line.startswith(("torch", "tqdm"))
    }
    assert len(columns) == 1
    replaced = ManifestText(text)
    replaced.put(_PYTHON, "torch", spec=">=3.0")
    assert 'torch     = ">=3.0"' in replaced.text()


def test_a_table_the_manifest_never_had_is_written_as_one_heading(text: str) -> None:
    """A new ecosystem reads as `[rust.deps]`, the way every other table is written."""
    edited = ManifestText(text)
    edited.put(("rust", "deps"), "ripgrep", spec=">=14, <15")
    written = edited.text()
    assert "[rust.deps]" in written
    assert 'ripgrep = ">=14, <15"' in written
    assert "\n[rust]\n" not in written


def test_dropping_the_last_requirement_takes_the_table_it_left_empty_with_it(text: str) -> None:
    """`put` writes a table the manifest never had, so `drop` must be able to unwrite it."""
    edited = ManifestText(text)
    edited.put(("rust", "deps"), "ripgrep", spec=">=14, <15")
    edited.put(("rust", "deps"), "fd", spec=">=10")
    edited.drop(("rust", "deps"), "fd")
    assert "[rust.deps]" in edited.text()
    edited.drop(("rust", "deps"), "ripgrep")
    assert edited.text() == text
    edited.drop(_PYTHON, "torch")
    assert "[python.deps]" in edited.text()  # the table still declares lab-core


@pytest.mark.parametrize(
    ("name", "expected"),
    [("torch", ">=2.9"), ("lab-core", '{ path = "packages/lab-core", editable = true }')],
)
def test_constraint_reports_a_version_or_the_source_standing_in_for_one(
    text: str, name: str, expected: str
) -> None:
    """A requirement carrying a source has no range, so it answers with how it is written."""
    assert ManifestText(text).constraint(_PYTHON, name) == expected


@pytest.mark.parametrize(
    ("path", "name", "present"),
    [
        (_PYTHON, "torch", True),
        (_PYTHON, "absent", False),
        (("rust", "deps"), "torch", False),
        (("workspace", "name"), "torch", False),
    ],
)
def test_declares_answers_for_a_missing_key_table_or_branch(
    text: str, path: tuple[str, ...], name: str, present: bool
) -> None:
    """Asking about a table that is not there is a question with an answer, not a failure."""
    assert ManifestText(text).declares(path, name) is present


@pytest.mark.parametrize(
    ("path", "match"),
    [
        (("rust", "deps"), r"\[rust\] is not in this manifest"),
        (("workspace", "name"), "not a table of requirements"),
    ],
)
def test_reaching_a_table_that_is_not_there_names_the_heading_it_wanted(
    text: str, path: tuple[str, ...], match: str
) -> None:
    """A missing table and a key path landing on a value are both mistakes worth naming."""
    with pytest.raises(MissionError, match=match):
        ManifestText(text).table(path)
